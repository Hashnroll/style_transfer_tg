#!/usr/bin/env python
# -*- coding: utf-8 -*-
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
import os
from utils import load_img
from torchvision.utils import save_image
from model import encoders, decoders
from wct import WCT


TOKEN = os.environ['BOT_TOKEN']
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)
keyboard_markup = types.ReplyKeyboardMarkup()
btns_text = ('Загрузить содержание', 'Загрузить стиль', 'Применить стиль')
for text in btns_text:
    keyboard_markup.row(types.KeyboardButton(text))


model = WCT(0.25, encoders, decoders)
model.eval()


@dp.message_handler(commands=['start'])
async def start_cmd_handler(message: types.Message) -> None:
    await message.reply("Привет! Я StylusBot - умею применять любой стиль к любой картинке.\n"
                        "Для загрузки содержания и стиля воспользуйтесь меню. "
                        "А затем нажмите кнопку 'Применить стиль'.", reply_markup=keyboard_markup)

TYPE = 'content'
@dp.message_handler(text='Загрузить содержание')
async def all_msg_handler(message: types.Message):
    await message.reply('Загрузите картинку', reply_markup=keyboard_markup)
    global TYPE
    TYPE = 'content'


@dp.message_handler(text='Загрузить стиль')
async def all_msg_handler(message: types.Message):
    await message.reply('Загрузите стиль', reply_markup=keyboard_markup)
    global TYPE
    TYPE = 'style'


@dp.message_handler(text='Применить стиль')
async def all_msg_handler(message: types.Message):
    await message.reply('Применяю стиль...', reply_markup=keyboard_markup)
    await types.ChatActions.upload_photo()
    img_content = load_img('content.jpg').to('cuda')
    img_style = load_img('style.jpg', new_size=256).to('cuda')
    result = model(img_content.unsqueeze(0), img_style.unsqueeze(0))
    save_image(result, 'result.jpg')
    await bot.send_photo(message.chat.id, open('result.jpg', 'rb'))


@dp.message_handler(content_types=['photo'])
async def handle_photo(message):
    await message.photo[-1].download(f'{TYPE}.jpg')


if __name__ == '__main__':
    executor.start_polling(dp)