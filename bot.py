import os
import logging
import io
import tempfile
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
import pdfplumber
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Configuration ---
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Initialize Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')

# In-memory storage for user documents
# Format: {user_id: {"text": ..., "vectorstore": ..., "filename": ...}}
user_docs = {}

# Max file size: 20 MB (Telegram limit for bots)
MAX_FILE_SIZE = 20 * 1024 * 1024


def extract_text_from_pdf(pdf_bytes):
    """Extract text from PDF bytes using pdfplumber."""
    text = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def create_vector_store(text, user_id):
    """Create a FAISS vector store from text chunks."""
    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.split_text(text)

    # Create documents
    documents = [Document(page_content=chunk, metadata={"source": user_id}) for chunk in chunks]

    # Create embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Welcome message."""
    user = update.effective_user
    await update.message.reply_text(
        f"👋 Hello {user.first_name}!\n\n"
        f"Welcome to **DocMind** — your PDF assistant.\n\n"
        f"**How to use:**\n"
        f"1️⃣ Send me a PDF file\n"
        f"2️⃣ I'll process it and let you know when ready\n"
        f"3️⃣ Ask me anything about the document\n\n"
        f"**Features:**\n"
        f"📝 Generate summary\n"
        f"🔍 Search inside PDF\n"
        f"💬 Ask questions\n"
        f"📌 Extract key points\n\n"
        f"Send a PDF to get started!",
        parse_mode="Markdown"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Help message."""
    await update.message.reply_text(
        "🆘 **DocMind Help**\n\n"
        "**Commands:**\n"
        "/start - Welcome menu\n"
        "/help - This message\n"
        "/summary - Get a summary of your PDF\n"
        "/points - Extract key points\n"
        "/clear - Remove your current document\n\n"
        "**How to use:**\n"
        "1. Send a PDF file\n"
        "2. Wait for processing confirmation\n"
        "3. Ask any question about the document\n\n"
        "**Limits:**\n"
        "• Max file size: 20 MB\n"
        "• One document at a time per user",
        parse_mode="Markdown"
    )


async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle PDF uploads."""
    user_id = update.effective_user.id
    document = update.message.document

    # Check file type
    if not document.file_name.lower().endswith('.pdf'):
        await update.message.reply_text("❌ Please send a PDF file.")
        return

    # Check file size
    if document.file_size > MAX_FILE_SIZE:
        await update.message.reply_text(
            f"❌ File too large. Maximum size is 20 MB.\n"
            f"Your file: {document.file_size / 1024 / 1024:.1f} MB"
        )
        return

    # Send processing message
    processing_msg = await update.message.reply_text("📄 Processing your PDF...\n⏳ This may take a moment.")

    try:
        # Download file
        file = await document.get_file()
        pdf_bytes = await file.download_as_bytearray()

        # Extract text
        text = extract_text_from_pdf(pdf_bytes)

        if not text.strip():
            await processing_msg.edit_text(
                "❌ Could not extract text from this PDF.\n"
                "It may be a scanned document or contain only images."
            )
            return

        # Create vector store
        await processing_msg.edit_text("🔍 Analyzing document...")
        vectorstore = create_vector_store(text, user_id)

        # Store for user
        user_docs[user_id] = {
            "text": text,
            "vectorstore": vectorstore,
            "filename": document.file_name,
            "chunks": len(text) // 1000 + 1
        }

        # Success message with menu
        keyboard = [
            [InlineKeyboardButton("📝 Summary", callback_data="summary")],
            [InlineKeyboardButton("📌 Key Points", callback_data="points")],
            [InlineKeyboardButton("🔍 Search", callback_data="search_help")],
            [InlineKeyboardButton("🗑️ Clear", callback_data="clear")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await processing_msg.edit_text(
            f"✅ **PDF processed!**\n\n"
            f"📄 File: `{document.file_name}`\n"
            f"📊 Text length: {len(text):,} characters\n"
            f"🧩 Chunks: {len(text) // 1000 + 1}\n\n"
            f"**Now you can:**\n"
            f"• Ask questions about the document\n"
            f"• Use the buttons below for quick actions",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    except Exception as e:
        logger.error(f"PDF processing error: {e}")
        await processing_msg.edit_text(
            "❌ Failed to process the PDF.\n"
            "Please make sure it's a valid text-based PDF and try again."
        )


async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Answer questions about the uploaded PDF."""
    user_id = update.effective_user.id
    question = update.message.text.strip()

    if user_id not in user_docs:
        await update.message.reply_text(
            "📄 Please upload a PDF first before asking questions."
        )
        return

    # Send typing indicator
    await update.message.chat.send_action(action="typing")

    try:
        doc_data = user_docs[user_id]
        vectorstore = doc_data["vectorstore"]

        # Search for relevant chunks
        results = vectorstore.similarity_search(question, k=4)

        if not results:
            await update.message.reply_text(
                "🔍 I couldn't find relevant information in the document for that question."
            )
            return

        # Build context
        context_text = "\n\n".join([doc.page_content for doc in results])

        # Generate answer using Gemini
        prompt = f"""Based on the following document excerpts, answer the user's question.
If the answer is not in the excerpts, say "I couldn't find that information in the document."

DOCUMENT EXCERPTS:
{context_text}

USER QUESTION: {question}

Answer:"""

        response = model.generate_content(prompt)
        answer = response.text

        await update.message.reply_text(
            f"💬 **Answer:**\n\n{answer}",
            parse_mode="Markdown"
        )

    except Exception as e:
        logger.error(f"Q&A error: {e}")
        await update.message.reply_text(
            "❌ Sorry, I couldn't process that question. Please try again."
        )


async def summary_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generate a summary of the uploaded PDF."""
    user_id = update.effective_user.id

    if user_id not in user_docs:
        await update.message.reply_text("📄 Please upload a PDF first.")
        return

    await update.message.chat.send_action(action="typing")

    try:
        text = user_docs[user_id]["text"][:15000]  # Limit for API

        prompt = f"""Provide a concise summary of the following document.
Focus on the main ideas, key findings, and important conclusions.

DOCUMENT:
{text}

Summary:"""

        response = model.generate_content(prompt)

        await update.message.reply_text(
            f"📝 **Summary:**\n\n{response.text}",
            parse_mode="Markdown"
        )

    except Exception as e:
        logger.error(f"Summary error: {e}")
        await update.message.reply_text("❌ Failed to generate summary.")


async def points_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Extract key points from the PDF."""
    user_id = update.effective_user.id

    if user_id not in user_docs:
        await update.message.reply_text("📄 Please upload a PDF first.")
        return

    await update.message.chat.send_action(action="typing")

    try:
        text = user_docs[user_id]["text"][:15000]

        prompt = f"""Extract the most important points from the following document.
Present them as a bullet list of 5-10 key takeaways.

DOCUMENT:
{text}

Key Points:"""

        response = model.generate_content(prompt)

        await update.message.reply_text(
            f"📌 **Key Points:**\n\n{response.text}",
            parse_mode="Markdown"
        )

    except Exception as e:
        logger.error(f"Points error: {e}")
        await update.message.reply_text("❌ Failed to extract key points.")


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clear the user's current document."""
    user_id = update.effective_user.id

    if user_id in user_docs:
        del user_docs[user_id]
        await update.message.reply_text("🗑️ Document cleared. Send a new PDF to start over.")
    else:
        await update.message.reply_text("📄 No document to clear.")


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle button clicks."""
    query = update.callback_query
    await query.answer()

    if query.data == "summary":
        await summary_command(update, context)
    elif query.data == "points":
        await points_command(update, context)
    elif query.data == "search_help":
        await query.edit_message_text(
            "🔍 **Search:**\n\n"
            "Simply type your question and I'll search the document for the answer.",
            parse_mode="Markdown"
        )
    elif query.data == "clear":
        user_id = query.from_user.id
        if user_id in user_docs:
            del user_docs[user_id]
        await query.edit_message_text("🗑️ Document cleared.")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log errors."""
    logger.error("Exception while handling an update:", exc_info=context.error)


def main() -> None:
    """Start the bot."""
    if not BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN is not set!")
        return
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY is not set!")
        return

    application = Application.builder().token(BOT_TOKEN).build()

    # Command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("summary", summary_command))
    application.add_handler(CommandHandler("points", points_command))
    application.add_handler(CommandHandler("clear", clear_command))

    # Callback handler
    application.add_handler(CallbackQueryHandler(button_callback))

    # PDF handler
    application.add_handler(MessageHandler(filters.Document.PDF, handle_pdf))

    # Text handler (questions)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))

    application.add_error_handler(error_handler)

    logger.info("Starting @DocMindBot with long polling...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
