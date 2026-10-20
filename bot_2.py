import os
from pathlib import Path
import discord
from discord.ext import commands
from dotenv import load_dotenv
import re  # 正規表現モジュールを追加


env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)
TOKEN = os.getenv("DISCORD_TOKEN")

# 1. Botの基本設定
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# 2. 判定パターンの設定（正規表現）
# 例: 'バ.*カ' は「バ」で始まり「カ」で終わる、間に何があってもOKという意味
NG_PATTERNS = [
    r"バ.*カ", 
    r"あ.*ほ", 
    r"う.*ざ.*い", 
    r"し.*ね"

    "@everyone", "valo", "OW", "ow", "ヴ", "コンパス", "ごみ", "ゴミ", "クソ", "くそ", "死ね", "しね", "ばか", "バカ", "うざい", "ウザい"
]

@bot.event
async def on_ready():
    print(f'ログインしました: {bot.user.name}')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # メッセージの中にパターンに一致するものがあるかチェック
    for pattern in NG_PATTERNS:
        # re.search を使うと、ワイルドカード的な判定ができる
        if re.search(pattern, message.content):
            await message.reply(f"{message.author.mention}さん、インキャで草wwwwww")
            break

    await bot.process_commands(message)

bot.run(TOKEN)