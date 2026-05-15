import os
from pathlib import Path
import discord
from discord.ext import commands
import ollama
from dotenv import load_dotenv

# 1. 環境変数の読み込み（修正箇所）
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)
TOKEN = os.getenv("DISCORD_TOKEN")

# 2. Botの基本設定
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

MODEL = 'llama3.2:3b'
# 非同期クライアントを一度だけ生成して使い回す
ollama_client = ollama.AsyncClient()

# 反応させたいNGワード
NG_WORDS = ["ちんこ", "まんこ", "ちんちん", "おめこ", "せっくす", "エロ", "おてぃんてぃん", "fuck"]

async def generate_tsukkomi(text):
    """風紀委員としてのツッコミを生成"""
    prompt = f"「{text}」という不適切な言葉を使ったユーザーに、風紀委員として20文字以内で冷たく鋭いツッコミを入れなさい。挨拶や解説は不要です。"
    
    try:
        response = await ollama_client.generate(model=MODEL, prompt=prompt)
        return response['response'].strip()
    except Exception as e:
        print(f"Ollama Error: {e}")
        return "その言葉、聞き捨てなりませんね。慎みなさい。"

@bot.event
async def on_ready():
    print(f'-----------------------------------------')
    print(f'風紀委員Bot 稼働中（NGワード検知モード）')
    print(f'ログイン: {bot.user}')
    print(f'-----------------------------------------')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # 判定用に小文字に変換（FUCKなどを検知するため）
    content_lower = message.content.lower()

    # 1. NGワードチェック
    is_ng = any(word in content_lower for word in NG_WORDS)

    if is_ng:
        # AIによるツッコミ生成
        tsukkomi = await generate_tsukkomi(message.content)
        
        if tsukkomi and "SAFE" not in tsukkomi.upper():
            await message.reply(tsukkomi)

    await bot.process_commands(message)

# 最後に実行
if TOKEN:
    bot.run(TOKEN)
else:
    print("エラー: DISCORD_TOKENが設定されていません。.envファイルを確認してください。")