import discord
from discord.ext import commands
from tkn import TOKEN
from ai_logic import get_class_summer
from ai_logic import get_class_winter
from ai_logic import get_class_autumn
from ai_logic import get_class_spring

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix='/', intents=intents)

@bot.event
async def on_ready():
    print(f'Бот подключен {bot.user}')

@bot.command('check')
async def check(ctx): #добавить model
    if ctx.message.attachments:
        for attachment in ctx.message.attachments:
            file_name = attachment.filename
            await attachment.save(f"./images/{file_name}")
        #await ctx.send("Картинка(и) сохренена(ы)")
        await ctx.send(get_class_summer(f"./images/{file_name}"))
    else:  
        await ctx.send("Вы забыли сохранить картинки")

@bot.command()
async def time(ctx):
    if ctx.message.content == "Лето":
        await ctx.send(get_class_summer)
    elif ctx.message.content == "Зима":
        await ctx.send(get_class_winter)
    elif ctx.message.content == "Осень":
        await ctx.send(get_class_autumn)
    elif ctx.message.content == "Весна":
        await ctx.send(get_class_spring)

@bot.command()
async def help(ctx):
    await ctx.send(f'Список команд для бота:\n1. /check - проверяет, загрузил ли пользователь картинку\n2. /time - какое время года выбрал пользователь')

bot.run(TOKEN)