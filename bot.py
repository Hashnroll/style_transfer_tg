#!/usr/bin/env python
# -*- coding: utf-8 -*-
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
from utils import load_img
from torchvision.utils import save_image
from model import encoders, decoders
from wct import WCT


TOKEN = '1008589830:AAG5Ry-feh_y_ejU27Qbp4KI637KZJ4iK6Q'
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)
keyboard_markup = types.ReplyKeyboardMarkup()
btns_text = ('Загрузить содержание', 'Загрузить стиль', 'Перенести стиль')
for text in btns_text:
    keyboard_markup.row(types.KeyboardButton(text))


model = WCT(0.25, encoders, decoders)
model.eval()

TYPE = None


@dp.message_handler(commands=['start'])
async def start_cmd_handler(message: types.Message) -> None:
    await message.reply("Привет, я StylusBot - могу перенести стиль с одной картинки на другую!\n"
                        "Для загрузки содержания и стиля воспользуйтесь меню. "
                        "А затем нажмите кнопку 'Применить стиль'.\n"
                        "Дополнительные функции можно посмотреть с помощью /help.", reply_markup=keyboard_markup)


@dp.message_handler(commands=['help'])
async def help_cmd_handler(message: types.Message) -> None:
    await message.reply("\\style X - изменить степень стилизации. "
                        "Чем больше X, тем больше внимания будет уделено стилю "
                        "и меньше содержанию. X - от 0 до 100. Начальное значение X - 25.",
                        reply_markup=keyboard_markup)


@dp.message_handler(commands=['style'])
async def style_cmd_handler(message: types.Message) -> None:
    new_alpha = int(message.text.split()[-1])
    if new_alpha < 0 or new_alpha > 100:
        await message.reply("Не могу применить данное значение стиля. "
                            "Оно должно быть от 0 до 100.", reply_markup=keyboard_markup)
    else:
        old_alpha = model.alpha
        model.alpha = new_alpha
        await message.reply(f"Изменяю значение стиля с {old_alpha} на {new_alpha}.", reply_markup=keyboard_markup)


@dp.message_handler(text='Загрузить содержание')
async def all_msg_handler(message: types.Message):
    await message.reply('Хорошо! Отправьте мне картинку с содержанием.', reply_markup=keyboard_markup)
    global TYPE
    TYPE = 'content'


@dp.message_handler(text='Загрузить стиль')
async def all_msg_handler(message: types.Message):
    await message.reply('Хорошо! Отправьте мне картинку со стилем.', reply_markup=keyboard_markup)
    global TYPE
    TYPE = 'style'


@dp.message_handler(text='Перенести стиль')
async def all_msg_handler(message: types.Message):
    await message.reply('Переношу стиль...', reply_markup=keyboard_markup)
    await types.ChatActions.upload_photo()
    img_content = load_img('content.jpg').to('cuda')
    img_style = load_img('style.jpg', new_size=256).to('cuda')
    result = model(img_content.unsqueeze(0), img_style.unsqueeze(0))
    save_image(result, 'result.jpg')
    await bot.send_photo(message.chat.id, open('result.jpg', 'rb'))


@dp.message_handler(content_types=['photo'])
async def handle_photo(message):
    await message.photo[-1].download(f'{TYPE}.jpg')
    if TYPE == 'content':
        await message.reply("Отлично, содержание загружено!")
    if TYPE == 'style':
        await message.reply("Отлично, стиль загружен!")


if __name__ == '__main__':
    executor.start_polling(dp)