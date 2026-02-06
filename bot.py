import asyncio
import aiohttp
import logging
import os
import urllib.parse
import time
import re
import mimetypes
from typing import Optional, Dict, Any, List, Tuple
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes, CommandHandler, Defaults
from telegram.constants import ParseMode
from telegram.error import RetryAfter, TimedOut, NetworkError
from dotenv import load_dotenv
from aiohttp import web
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
from pyrogram import Client
from pyrogram.errors import FloodWait

# Load environment variables from .env file
load_dotenv()

# Logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Disable httpx and telegram library verbose logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("telegram.ext").setLevel(logging.WARNING)
logging.getLogger("pyrogram").setLevel(logging.WARNING)

# Config from environment variables
BOT_TOKEN = os.getenv("BOT_TOKEN")
TERABOX_API = os.getenv("TERABOX_API", "")

# Pyrogram API credentials
API_ID = int(os.getenv("API_ID", "0"))
API_HASH = os.getenv("API_HASH", "")

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "")
DB_NAME = os.getenv("DB_NAME", "viralbox_db")

# Channels
TELEGRAM_CHANNEL_ID = int(os.getenv("TELEGRAM_CHANNEL_ID", ""))
RESULT_CHANNEL_ID = int(os.getenv("RESULT_CHANNEL_ID", ""))

# Worker URL Base
WORKER_URL_BASE = os.getenv("WORKER_URL_BASE", "")

# Channel Username for Watermark
CHANNEL_USERNAME = os.getenv("CHANNEL_USERNAME", "@Mid_Night_Hub")

# Webhook Configuration for Koyeb
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
PORT = int(os.getenv("PORT", "8000"))

# Aria2 Configuration - IMPROVED
ARIA2_RPC_URL = os.getenv("ARIA2_RPC_URL", "http://localhost:6800/jsonrpc")
ARIA2_SECRET = os.getenv("ARIA2_SECRET", "")  # Now required!
DOWNLOAD_DIR = os.getenv("DOWNLOAD_DIR", "/tmp/aria2_downloads")

# Validate Aria2 Secret
if not ARIA2_SECRET:
    import secrets
    ARIA2_SECRET = secrets.token_urlsafe(32)
    logger.warning(f"⚠️ ARIA2_SECRET not set! Generated random secret: {ARIA2_SECRET}")

# Terabox domains
TERABOX_DOMAINS = [
    "terabox.com", "1024terabox.com", "teraboxapp.com", "teraboxlink.com",
    "terasharelink.com", "terafileshare.com", "1024tera.com", "1024tera.cn",
    "teraboxdrive.com", "dubox.com"
]

# API timeout - increased for slow Terabox API
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "120"))

# File size limit (2GB for Pyrogram)
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "2000"))  # 2GB default

# Download progress update interval (seconds)
PROGRESS_UPDATE_INTERVAL = 10

# Validate required environment variables
if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN environment variable is required!")
if not API_ID or not API_HASH:
    raise ValueError("API_ID and API_HASH are required for Pyrogram!")

# --- Pyrogram Client Setup ---
pyrogram_client = None

async def init_pyrogram():
    global pyrogram_client
    try:
        # Use /tmp for session file (writable directory)
        session_path = "/tmp/pyrogram_session"
        os.makedirs(session_path, exist_ok=True)
        
        pyrogram_client = Client(
            "terabox_bot",
            api_id=API_ID,
            api_hash=API_HASH,
            bot_token=BOT_TOKEN,
            workdir=session_path
        )
        await pyrogram_client.start()
        logger.info("✅ Pyrogram client started successfully")
    except Exception as e:
        logger.error(f"❌ Pyrogram client failed to start: {e}")
        raise

async def stop_pyrogram():
    global pyrogram_client
    if pyrogram_client:
        try:
            await pyrogram_client.stop()
            logger.info("🛑 Pyrogram client stopped")
        except Exception as e:
            logger.error(f"Error stopping Pyrogram: {e}")

# --- MongoDB Database Setup ---
mongo_client = None
db = None
files_collection = None
mappings_collection = None

async def init_db():
    global mongo_client, db, files_collection, mappings_collection
    try:
        mongo_client = AsyncIOMotorClient(MONGO_URI)
        db = mongo_client[DB_NAME]
        files_collection = db["terabox_file_name"]
        mappings_collection = db["mappings"]
        
        # Create indexes for better performance
        await files_collection.create_index("file_name", unique=True)
        
        # Create index for mappings
        await mappings_collection.create_index("mapping", unique=True)
        await mappings_collection.create_index("message_id")
        
        logger.info("✅ MongoDB connected successfully")
    except Exception as e:
        logger.error(f"❌ MongoDB connection failed: {e}")
        raise

async def close_db():
    global mongo_client
    if mongo_client:
        try:
            mongo_client.close()
            logger.info("🛑 MongoDB connection closed")
        except Exception as e:
            logger.error(f"Error closing MongoDB: {e}")

async def is_file_processed(file_name: str) -> bool:
    try:
        result = await files_collection.find_one({"file_name": file_name})
        return result is not None
    except Exception as e:
        logger.error(f"Error checking file in DB: {e}")
        return False

async def save_file_info(file_name: str, file_size: str):
    try:
        document = {
            "file_name": file_name,
            "file_size": file_size
        }
        await files_collection.insert_one(document)
        logger.info(f"✅ Saved to DB: {file_name}")
    except Exception as e:
        logger.warning(f"Failed to save to DB (might be duplicate): {e}")

import random
import string

def generate_random_mapping(length: int = 6) -> str:
    """Generate random alphanumeric mapping string"""
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))

async def save_mapping(message_id: int) -> str:
    """Save message_id with random mapping and return the mapping"""
    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            mapping = generate_random_mapping()
            document = {
                "mapping": mapping,
                "message_id": message_id
            }
            await mappings_collection.insert_one(document)
            logger.info(f"✅ Saved mapping: {mapping} -> {message_id}")
            return mapping
        except Exception as e:
            if attempt == max_attempts - 1:
                logger.error(f"Failed to save mapping after {max_attempts} attempts: {e}")
                raise
            # Duplicate mapping, try again
            continue
    raise Exception("Failed to generate unique mapping")

# ---------------- Aria2Client - IMPROVED ----------------
class Aria2Client:
    def __init__(self, rpc_url: str, secret: str):
        self.rpc_url = rpc_url
        self.secret = secret
        self.session: Optional[aiohttp.ClientSession] = None

    async def init_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600))
            logger.info("✅ Aria2 RPC session initialized")

    async def close_session(self):
        if self.session:
            try:
                await self.session.close()
                self.session = None
                logger.info("🛑 Aria2 RPC session closed")
            except Exception as e:
                logger.error(f"Error closing Aria2 session: {e}")

    async def _call_rpc(self, method: str, params: list = None) -> Dict[str, Any]:
        if params is None:
            params = []
        if self.secret:
            params.insert(0, f"token:{self.secret}")
        payload = {"jsonrpc": "2.0", "id": f"aria2_{int(time.time())}", "method": method, "params": params}
        try:
            await self.init_session()
            async with self.session.post(self.rpc_url, json=payload) as r:
                result = await r.json()
                if "error" in result:
                    return {"success": False, "error": result["error"]}
                return {"success": True, "result": result.get("result")}
        except Exception as e:
            logger.error(f"Aria2 RPC call failed: {method} - {e}")
            return {"success": False, "error": str(e)}

    async def add_download(self, url: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        if options is None:
            options = {}
        # OPTIMIZED ARIA2 OPTIONS FOR FASTER DOWNLOADS
        opts = {
            "dir": DOWNLOAD_DIR,
            "continue": "true",
            "max-connection-per-server": "16",  # Increased from 10
            "split": "16",  # Increased from 10
            "min-split-size": "1M",
            "max-concurrent-downloads": "5",
            "max-download-limit": "0",  # No limit
            "lowest-speed-limit": "0",
            "disable-ipv6": "true",
            "connect-timeout": "30",
            "timeout": "30",
            "max-tries": "5",
            "retry-wait": "3",
        }
        opts.update(options)
        return await self._call_rpc("aria2.addUri", [[url], opts])

    async def get_status(self, gid: str) -> Dict[str, Any]:
        """Get download status with progress info"""
        result = await self._call_rpc("aria2.tellStatus", [gid, ["status", "totalLength", "completedLength", "downloadSpeed", "errorMessage"]])
        return result

    async def wait_for_download(self, gid: str, status_msg=None, link_number: int = 1, total_links: int = 1, file_name: str = ""):
        """Wait for download with progress updates"""
        last_update = 0
        
        while True:
            status = await self.get_status(gid)
            if not status["success"]:
                return status
            
            info = status["result"]
            current_status = info.get("status", "")
            
            if current_status == "complete":
                files_result = await self._call_rpc("aria2.getFiles", [gid])
                if files_result["success"]:
                    return {"success": True, "files": files_result["result"]}
                return {"success": False, "error": "Failed to get file info"}
                
            elif current_status in ["error", "removed"]:
                error_msg = info.get("errorMessage", "Download failed")
                return {"success": False, "error": error_msg}
            
            # Update progress every PROGRESS_UPDATE_INTERVAL seconds
            current_time = time.time()
            if status_msg and (current_time - last_update) >= PROGRESS_UPDATE_INTERVAL:
                try:
                    total = int(info.get("totalLength", 0))
                    completed = int(info.get("completedLength", 0))
                    speed = int(info.get("downloadSpeed", 0))
                    
                    if total > 0:
                        percent = (completed / total) * 100
                        speed_mb = speed / (1024 * 1024)
                        
                        # Format file name for display (decode URL encoding)
                        display_name = urllib.parse.unquote(file_name)
                        
                        await status_msg.edit_text(
                            f"🔄 Processing link {link_number}/{total_links}...\n"
                            f"📦 {display_name}\n"
                            f"⬇️ Downloading...\n"
                            f"⬇️ GID: {gid[:8]}... | {percent:.1f}%\n"
                            f"⚡ Speed: {speed_mb:.2f} MB/s",
                            parse_mode=ParseMode.HTML
                        )
                        last_update = current_time
                except Exception as e:
                    logger.warning(f"Failed to update progress: {e}")
            
            await asyncio.sleep(2)

# ---------------- Global Task Queue ----------------
class GlobalTaskQueue:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.processing = False
        self.worker_task = None
        
    async def start_worker(self):
        """Start the background worker that processes tasks one by one"""
        if self.worker_task is None or self.worker_task.done():
            self.worker_task = asyncio.create_task(self._worker())
            logger.info("🚀 Global task queue worker started")
    
    async def stop_worker(self):
        """Stop the background worker"""
        if self.worker_task and not self.worker_task.done():
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                logger.info("🛑 Queue worker stopped")
    
    async def _worker(self):
        """Background worker that processes queue items one by one"""
        while True:
            try:
                # Get next task from queue (wait if empty)
                task = await self.queue.get()
                
                self.processing = True
                logger.info(f"📋 Queue size: {self.queue.qsize()} | Processing new task")
                
                # Process the task
                try:
                    await task["func"](**task["kwargs"])
                except Exception as e:
                    logger.error(f"Task execution failed: {e}", exc_info=True)
                
                # Mark task as done
                self.queue.task_done()
                self.processing = False
                
                # Small delay between tasks
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                logger.info("Queue worker cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Error in queue worker: {e}", exc_info=True)
                self.processing = False
    
    async def add_task(self, func, **kwargs):
        """Add a task to the queue"""
        await self.queue.put({"func": func, "kwargs": kwargs})
        logger.info(f"➕ Task added to queue. Queue size: {self.queue.qsize()}")

# Global queue instance
global_queue = GlobalTaskQueue()

# ---------------- Bot Logic ----------------
class TeraboxTelegramBot:
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.aria2 = Aria2Client(ARIA2_RPC_URL, ARIA2_SECRET)
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    async def init_session(self):
        if not self.session:
            # Increased timeout for slow Terabox API
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120))

    async def close_session(self):
        if self.session:
            try:
                await self.session.close()
                self.session = None
                logger.info("🛑 Bot HTTP session closed")
            except Exception as e:
                logger.error(f"Error closing bot session: {e}")

    def is_terabox_url(self, url: str) -> bool:
        try:
            domain = urllib.parse.urlparse(url).netloc.lower().removeprefix("www.")
            return domain in TERABOX_DOMAINS or any(d in domain for d in TERABOX_DOMAINS)
        except Exception as e:
            logger.error(f"Error parsing URL: {e}")
            return False

    def add_watermark_to_filename(self, original_name: str) -> str:
        """Add watermark before the original filename"""
        name, ext = os.path.splitext(original_name)
        return f"Telegram - {CHANNEL_USERNAME} {name}{ext}"

    async def download_from_terabox(self, url: str, max_retries: int = 3) -> Dict[str, Any]:
        """Download from Terabox with retry logic and proper timeout handling"""
        await self.init_session()
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Terabox API call attempt {attempt + 1}/{max_retries}")
                
                # API ko proper URL format chahiye
                api_url = f"{TERABOX_API}?url={urllib.parse.quote(url)}"
                
                # Increased timeout for slow API - 120 seconds
                async with self.session.get(api_url, timeout=aiohttp.ClientTimeout(total=120)) as r:
                    # Check if response is OK
                    if r.status != 200:
                        logger.warning(f"⚠️ API returned status {r.status}")
                        if attempt == max_retries - 1:
                            return {"success": False, "error": f"API returned status {r.status}"}
                        await asyncio.sleep(3)
                        continue
                    
                    data = await r.json()
                    
                    # Check for successful response
                    if data.get("status") == "✅ Successfully":
                        logger.info(f"✅ Terabox API success")
                        return {"success": True, "data": data}
                    else:
                        logger.warning(f"⚠️ Terabox API unsuccessful: {data.get('status', 'Unknown')}")
                        if attempt == max_retries - 1:
                            return {"success": False, "error": data.get("status", "Unknown error")}
                        await asyncio.sleep(3)
                        
            except asyncio.TimeoutError:
                logger.warning(f"⏱️ Timeout on attempt {attempt + 1} (API is slow)")
                if attempt == max_retries - 1:
                    return {"success": False, "error": "API timeout - try again later"}
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"❌ Error on attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    return {"success": False, "error": str(e)}
                await asyncio.sleep(3)
        
        return {"success": False, "error": "Unknown error"}

    async def upload_to_telegram_pyrogram(self, file_path: str, caption: str, max_retries: int = 5) -> Dict[str, Any]:
        """Upload to Telegram using Pyrogram (supports large files up to 2GB)"""
        for attempt in range(max_retries):
            try:
                # Determine file type based on mime type
                mime_type, _ = mimetypes.guess_type(file_path)
                
                # Get file size for progress
                file_size = os.path.getsize(file_path)
                logger.info(f"📤 Uploading file: {os.path.basename(file_path)} ({file_size / (1024*1024):.2f} MB)")
                
                # Upload based on file type
                if mime_type and mime_type.startswith("video"):
                    msg = await pyrogram_client.send_video(
                        chat_id=TELEGRAM_CHANNEL_ID,
                        video=file_path,
                        caption=caption,
                        progress=self._upload_progress
                    )
                elif mime_type and mime_type.startswith("image"):
                    msg = await pyrogram_client.send_photo(
                        chat_id=TELEGRAM_CHANNEL_ID,
                        photo=file_path,
                        caption=caption,
                        progress=self._upload_progress
                    )
                else:
                    msg = await pyrogram_client.send_document(
                        chat_id=TELEGRAM_CHANNEL_ID,
                        document=file_path,
                        caption=caption,
                        progress=self._upload_progress
                    )
                
                # Extract file_id
                file_id = None
                if msg.video:
                    file_id = msg.video.file_id
                elif msg.photo:
                    file_id = msg.photo.file_id
                elif msg.document:
                    file_id = msg.document.file_id
                
                logger.info(f"✅ Upload successful - Message ID: {msg.id}")
                return {"success": True, "message_id": msg.id, "file_id": file_id}
                
            except FloodWait as e:
                wait_time = e.value + 2
                logger.warning(f"⏳ FloodWait: Waiting {wait_time}s")
                await asyncio.sleep(wait_time)
            except Exception as e:
                logger.error(f"❌ Upload error (attempt {attempt + 1}): {str(e)}")
                if attempt == max_retries - 1:
                    return {"success": False, "error": str(e)}
                await asyncio.sleep(5)
        
        return {"success": False, "error": "Upload failed after retries"}
    
    def _upload_progress(self, current, total):
        """Progress callback for Pyrogram upload"""
        percentage = (current / total) * 100
        if int(percentage) % 10 == 0:  # Log every 10%
            logger.info(f"📤 Upload progress: {percentage:.1f}% ({current / (1024*1024):.2f} MB / {total / (1024*1024):.2f} MB)")

# ---------------- Task Processing Functions ----------------
bot_instance = TeraboxTelegramBot()

async def process_single_link(url: str, link_number: int, total_links: int, 
                              context: ContextTypes.DEFAULT_TYPE, status_msg) -> Tuple[bool, Optional[Dict], Optional[str]]:
    """Process a single link with improved error handling"""
    file_path = None
    try:
        # Step 1: Get file info
        await status_msg.edit_text(
            f"🔄 Processing link {link_number}/{total_links}...\n"
            f"🔍 Fetching file info...",
            parse_mode=ParseMode.HTML
        )

        result = await bot_instance.download_from_terabox(url)
        if not result["success"]:
            error_msg = f"❌ API Error: {result['error']}"
            logger.error(error_msg)
            return False, None, error_msg

        data = result["data"]
        original_file_name = data.get("file_name", "unknown_file")
        file_size_str = data.get("file_size", "Unknown")
        
        # Decode filename if URL encoded
        try:
            original_file_name = urllib.parse.unquote(original_file_name)
        except Exception:
            pass

        watermarked_name = bot_instance.add_watermark_to_filename(original_file_name)

        # Check if already processed
        if await is_file_processed(original_file_name):
            logger.info(f"⏭️ Skipping duplicate: {original_file_name}")
            return False, None, f"⏭️ Already processed"

        # Parse file size and check limit
        try:
            size_mb = 0
            if "MB" in file_size_str:
                size_mb = float(file_size_str.replace("MB", "").strip())
            elif "GB" in file_size_str:
                size_mb = float(file_size_str.replace("GB", "").strip()) * 1024
            
            if size_mb > MAX_FILE_SIZE_MB:
                error_msg = f"❌ File too large: {file_size_str} (max {MAX_FILE_SIZE_MB}MB)"
                logger.info(error_msg)
                return False, None, error_msg
        except (ValueError, AttributeError) as e:
            logger.warning(f"Could not parse file size: {file_size_str} - {e}")

        # Step 2: Download with progress
        dl_url = data.get("streaming_url") or data.get("download_link")
        if not dl_url:
            error_msg = f"❌ No download link available"
            logger.error(error_msg)
            return False, None, error_msg

        logger.info(f"📥 Starting download: {original_file_name}")
        dl = await bot_instance.aria2.add_download(dl_url, {"out": watermarked_name})
        
        if not dl["success"]:
            error_msg = f"❌ Download init failed: {dl.get('error', 'Unknown')}"
            logger.error(error_msg)
            return False, None, error_msg

        gid = dl["result"]
        logger.info(f"✅ Download started: GID {gid}")
        
        # Wait for download with progress updates
        done = await bot_instance.aria2.wait_for_download(
            gid, 
            status_msg=status_msg, 
            link_number=link_number, 
            total_links=total_links,
            file_name=original_file_name
        )
        
        if not done["success"]:
            error_msg = f"❌ Download failed: {done.get('error', 'Unknown')}"
            logger.error(error_msg)
            return False, None, error_msg

        file_path = done["files"][0]["path"]
        logger.info(f"✅ Downloaded: {file_path}")

        # Step 3: Upload to Telegram
        await status_msg.edit_text(
            f"🔄 Processing link {link_number}/{total_links}...\n"
            f"📦 {original_file_name}\n"
            f"⬆️ Uploading to channel...",
            parse_mode=ParseMode.HTML
        )

        caption_file = f"📁 File Name: {watermarked_name}\n📊 File Size: {file_size_str}"
        
        upload_result = await bot_instance.upload_to_telegram_pyrogram(file_path, caption_file)
        
        if not upload_result["success"]:
            error_msg = f"❌ Upload failed: {upload_result.get('error', 'Unknown')}"
            logger.error(error_msg)
            return False, None, error_msg

        message_id = upload_result["message_id"]
        file_id = upload_result["file_id"]
        
        logger.info(f"✅ Uploaded - Msg ID: {message_id}")

        # Step 4: Save to DB
        await save_file_info(original_file_name, file_size_str)

        # Generate mapping and save to DB
        mapping = await save_mapping(message_id)
        
        # Build worker URL
        worker_url = f"{WORKER_URL_BASE}/{mapping}"
        
        # Result data
        result_data = {
            "original_name": original_file_name,
            "watermarked_name": watermarked_name,
            "file_size": file_size_str,
            "message_id": message_id,
            "file_id": file_id,
            "mapping": mapping,
            "worker_url": worker_url
        }

        logger.info(f"[{link_number}/{total_links}] ✅ Success: {original_file_name}")
        return True, result_data, None

    except Exception as e:
        error_msg = f"❌ Unexpected error: {str(e)}"
        logger.error(f"Error processing link: {str(e)}", exc_info=True)
        return False, None, error_msg
    
    finally:
        # Cleanup downloaded file
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"🗑️ Deleted: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete file: {e}")

async def process_task(urls: List[str], context: ContextTypes.DEFAULT_TYPE, 
                      message_id: int, chat_id: int, user_message_id: int, user_media_message):
    """Process all links from one user message - ONE BY ONE"""
    reply_msg = None
    try:
        total_links = len(urls)
        successful_results = []
        failed_links = []

        # Send initial reply
        reply_msg = await context.bot.send_message(
            chat_id=chat_id,
            reply_to_message_id=user_message_id,
            text=f"⏳ Processing {total_links} link(s)...",
            parse_mode=ParseMode.HTML
        )

        logger.info(f"📋 Processing {total_links} links sequentially")

        # Process each link ONE BY ONE
        for idx, url in enumerate(urls, 1):
            logger.info(f"▶️ Processing link {idx}/{total_links}")
            
            success, result_data, error_msg = await process_single_link(
                url, idx, total_links, context, reply_msg
            )

            if success and result_data:
                successful_results.append(result_data)
            elif error_msg:
                failed_links.append(f"Link {idx}: {error_msg}")
            
            # Small delay between links
            if idx < total_links:
                await asyncio.sleep(2)

        # Post to result channel if ANY successful results exist
        if successful_results:
            try:
                # Build caption with all successful worker URLs
                result_caption = ""
                for result in successful_results:
                    result_caption += f"✅ {result['worker_url']}\n"
                
                result_caption = result_caption.strip()
                
                # Copy user's media message to result channel with worker URLs
                await context.bot.copy_message(
                    chat_id=RESULT_CHANNEL_ID,
                    from_chat_id=chat_id,
                    message_id=user_message_id,
                    caption=result_caption,
                    parse_mode=ParseMode.HTML
                )
                logger.info(f"✅ Posted {len(successful_results)} successful results to result channel")
                
            except Exception as e:
                logger.error(f"Failed to copy to result channel: {e}", exc_info=True)

        # Update reply message with final status
        if failed_links and successful_results:
            # Some succeeded, some failed
            final_msg = f"⚠️ <b>Partial Success</b>\n\n"
            final_msg += f"✅ Successful: {len(successful_results)}\n"
            final_msg += f"❌ Failed: {len(failed_links)}\n\n"
            final_msg += "<b>Failed Links:</b>\n"
            final_msg += "\n".join(failed_links[:5])  # Show first 5 errors
            
            try:
                await reply_msg.edit_text(final_msg, parse_mode=ParseMode.HTML)
            except Exception as e:
                logger.warning(f"Failed to update message: {e}")
            
            logger.warning(f"⚠️ Task completed: {len(successful_results)} success, {len(failed_links)} failed")
            
            # Delete user's original message
            try:
                await context.bot.delete_message(chat_id=chat_id, message_id=user_message_id)
                logger.info(f"🗑️ Deleted user message")
            except Exception as e:
                logger.warning(f"Failed to delete user message: {e}")
                
        elif failed_links and not successful_results:
            # All failed
            error_summary = "❌ <b>All Links Failed</b>\n\n"
            error_summary += "\n".join(failed_links)
            
            try:
                await reply_msg.edit_text(error_summary, parse_mode=ParseMode.HTML)
            except Exception as e:
                logger.warning(f"Failed to update message: {e}")
            
            logger.warning(f"❌ Task failed: All {len(failed_links)} links failed")
            return
            
        else:
            # All succeeded
            try:
                await reply_msg.edit_text(
                    f"✅ <b>All {len(successful_results)} links processed successfully!</b>",
                    parse_mode=ParseMode.HTML
                )
                await asyncio.sleep(2)
            except Exception as e:
                logger.warning(f"Failed to update message: {e}")
            
            # Delete both messages on complete success
            try:
                await context.bot.delete_message(chat_id=chat_id, message_id=user_message_id)
                logger.info(f"🗑️ Deleted user message")
            except Exception as e:
                logger.warning(f"Failed to delete user message: {e}")
            
            try:
                await reply_msg.delete()
                logger.info(f"🗑️ Deleted reply message")
            except Exception as e:
                logger.warning(f"Failed to delete reply message: {e}")

            logger.info(f"✅ Task completed successfully: {len(successful_results)} files processed")

    except Exception as e:
        logger.error(f"❌ Error in process_task: {e}", exc_info=True)
        # On unexpected error, show error and keep messages
        if reply_msg:
            try:
                await reply_msg.edit_text(
                    f"❌ <b>Unexpected Error</b>\n\n{str(e)}",
                    parse_mode=ParseMode.HTML
                )
            except Exception as edit_error:
                logger.error(f"Failed to update error message: {edit_error}")

async def handle_media_with_links(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming media with links - ADD TO QUEUE (Ignore only configured channels)"""
    m = update.effective_message
    if not m:
        return

    # IGNORE ONLY CONFIGURED CHANNELS (TELEGRAM_CHANNEL_ID and RESULT_CHANNEL_ID)
    if m.chat.id == TELEGRAM_CHANNEL_ID or m.chat.id == RESULT_CHANNEL_ID:
        logger.info(f"⏭️ Ignoring post from configured channel: {m.chat.id}")
        return

    try:
        caption = m.caption or ""
        urls = re.findall(r"https?://[^\s]+", caption)
        urls = list(dict.fromkeys(urls))  # Remove duplicates

        if not urls:
            return

        terabox_links = [u for u in urls if bot_instance.is_terabox_url(u)]

        if not terabox_links:
            await m.reply_text(
                "❌ No Terabox links found.",
                parse_mode=ParseMode.HTML
            )
            return

        logger.info(f"📨 Received {len(terabox_links)} Terabox links from user")
        
        # Add to queue (will be processed one by one by global worker)
        await global_queue.add_task(
            process_task,
            urls=terabox_links,
            context=context,
            message_id=m.message_id,
            chat_id=m.chat_id,
            user_message_id=m.message_id,
            user_media_message=m
        )

    except Exception as e:
        logger.error(f"Error in handle_media_with_links: {e}", exc_info=True)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    m = update.effective_message
    if not m:
        return
    await m.reply_text(
        f"✅ <b>Bot is Running!</b>\n\n"
        f"📌 <b>Features:</b>\n"
        f"• <b>Pyrogram integration</b> - Upload files up to {MAX_FILE_SIZE_MB}MB!\n"
        f"• One-by-one processing (no overload)\n"
        f"• Global queue system\n"
        f"• Automatic file renaming with watermark\n"
        f"• MongoDB storage for duplicate detection\n"
        f"• Flood wait handling\n"
        f"• Ignores configured channels\n"
        f"• Optimized Aria2 downloads (16 connections)\n"
        f"• Real-time progress updates\n\n"
        f"📋 <b>How to use:</b>\n"
        f"Send media (photo/video/document) with Terabox links in caption.\n\n"
        f"💡 <b>Example:</b>\n"
        f"Send a photo with caption:\n"
        f"<code>https://terabox.com/s/xxxxx</code>",
        parse_mode=ParseMode.HTML
    )

# Health check endpoint
async def health_check(request):
    return web.Response(text="OK", status=200)

async def webhook_handler(request):
    """Handle incoming webhook updates"""
    try:
        data = await request.json()
        update = Update.de_json(data, application.bot)
        await application.process_update(update)
        return web.Response(status=200)
    except Exception as e:
        logger.error(f"Webhook error: {e}", exc_info=True)
        return web.Response(status=500)

# Global application instance
application = None

# CLEANUP HANDLER - NEW!
async def shutdown_handler(app: Application):
    """Cleanup on shutdown"""
    logger.info("🛑 Starting shutdown sequence...")
    
    try:
        # Stop queue worker
        await global_queue.stop_worker()
        
        # Close Aria2 session
        await bot_instance.aria2.close_session()
        
        # Close bot session
        await bot_instance.close_session()
        
        # Close MongoDB
        await close_db()
        
        # Stop Pyrogram
        await stop_pyrogram()
        
        logger.info("✅ Shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)

async def start_webhook_server():
    """Start the webhook server"""
    global application
    
    # Initialize database
    await init_db()
    
    # Initialize Pyrogram
    await init_pyrogram()
    
    # Initialize Aria2 session
    await bot_instance.aria2.init_session()
    
    # Create application
    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .defaults(Defaults(parse_mode=ParseMode.HTML))
        .build()
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(
        filters.PHOTO | filters.VIDEO | filters.Document.ALL,
        handle_media_with_links
    ))

    # Add shutdown handler
    application.post_shutdown = shutdown_handler

    # Initialize the application
    await application.initialize()
    await application.start()
    
    # Start global queue worker
    await global_queue.start_worker()

    # Set webhook
    if WEBHOOK_URL:
        webhook_path = f"/webhook/{BOT_TOKEN}"
        full_webhook_url = f"{WEBHOOK_URL}{webhook_path}"
        await application.bot.set_webhook(url=full_webhook_url)
        logger.info(f"Webhook set to: {full_webhook_url}")
    else:
        logger.warning("WEBHOOK_URL not set!")

    # Create web application
    app = web.Application()
    app.router.add_get("/health", health_check)
    app.router.add_get("/", health_check)
    app.router.add_post(f"/webhook/{BOT_TOKEN}", webhook_handler)

    # Start web server
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()

    logger.info(f"🚀 Bot started in webhook mode on port {PORT}")
    logger.info(f"📢 Telegram Channel: {TELEGRAM_CHANNEL_ID}")
    logger.info(f"📊 Result Channel: {RESULT_CHANNEL_ID}")
    logger.info(f"🏷️ Watermark: {CHANNEL_USERNAME}")
    logger.info(f"📋 Global queue system active")
    logger.info(f"⏭️ Ignoring channels: {TELEGRAM_CHANNEL_ID}, {RESULT_CHANNEL_ID}")
    logger.info(f"📤 Pyrogram enabled - Max file size: {MAX_FILE_SIZE_MB}MB")
    logger.info(f"🔐 Aria2 RPC secured with secret")
    logger.info(f"⚡ Aria2 optimized: 16 connections per server")

    # Keep the server running
    await asyncio.Event().wait()

def main():
    """Main entry point"""
    if WEBHOOK_URL or os.getenv("PORT"):
        logger.info("Starting in WEBHOOK mode")
        asyncio.run(start_webhook_server())
    else:
        logger.info("Starting in POLLING mode")
        
        async def run_polling():
            global application
            
            await init_db()
            
            # Initialize Pyrogram
            await init_pyrogram()
            
            # Initialize Aria2 session
            await bot_instance.aria2.init_session()
            
            application = (
                Application.builder()
                .token(BOT_TOKEN)
                .defaults(Defaults(parse_mode=ParseMode.HTML))
                .build()
            )

            application.add_handler(CommandHandler("start", start))
            application.add_handler(MessageHandler(
                filters.PHOTO | filters.VIDEO | filters.Document.ALL,
                handle_media_with_links
            ))
            
            # Add shutdown handler
            application.post_shutdown = shutdown_handler
            
            # Start global queue worker
            await global_queue.start_worker()

            logger.info("🚀 Bot started in polling mode")
            logger.info(f"📢 Telegram Channel: {TELEGRAM_CHANNEL_ID}")
            logger.info(f"📊 Result Channel: {RESULT_CHANNEL_ID}")
            logger.info(f"📋 Global queue system active")
            logger.info(f"⏭️ Ignoring channels: {TELEGRAM_CHANNEL_ID}, {RESULT_CHANNEL_ID}")
            logger.info(f"📤 Pyrogram enabled - Max file size: {MAX_FILE_SIZE_MB}MB")
            logger.info(f"🔐 Aria2 RPC secured with secret")
            logger.info(f"⚡ Aria2 optimized: 16 connections per server")
            
            await application.run_polling()
        
        asyncio.run(run_polling())

if __name__ == "__main__":
    main()
