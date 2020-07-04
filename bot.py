#!/usr/bin/env python
# -*- coding: utf-8 -*-
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.utils import executor
from aiofiles import os as aio_os
import os
from utils import load_img
from torchvision.utils import save_image
from torchvision import transforms
from model import encoders, decoders
from wct import WCT
from PIL import Image
from cyclegan.test_image import model as model_gan

TOKEN = '1008589830:AAG5Ry-feh_y_ejU27Qbp4KI637KZJ4iK6Q'
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)
keyboard_markup = types.ReplyKeyboardMarkup()
btns_text = ('Загрузить содержание', 'Загрузить стиль', 'Перенести стиль')
for text in btns_text:
    keyboard_markup.row(types.KeyboardButton(text))

inline_keyboard_markup = types.InlineKeyboardMarkup()
inline_keyboard_markup.add(types.InlineKeyboardButton('Применить стиль Сезанна', callback_data='sezanne'))

model = WCT(0.75, encoders, decoders)
model.eval()

TYPE = None


@dp.message_handler(commands=['start'])
async def start_cmd_handler(message: types.Message) -> None:
    if os.path.isfile(f'{message.from_user.id}/content.jpg'):
        os.remove(f'{message.from_user.id}/content.jpg')
    if os.path.isfile(f'{message.from_user.id}/style.jpg'):
        os.remove(f'{message.from_user.id}/style.jpg')
    if os.path.isfile(f'{message.from_user.id}/result.jpg'):
        os.remove(f'{message.from_user.id}/result.jpg')
    await message.reply("Привет, я StylusBot - могу перенести стиль с одной картинки на другую!\n"
                        "Для загрузки содержания и стиля воспользуйтесь меню. "
                        "А затем нажмите кнопку 'Применить стиль'.\n"
                        "Дополнительные функции можно посмотреть с помощью /help.", reply_markup=keyboard_markup)


@dp.message_handler(commands=['help'])
async def help_cmd_handler(message: types.Message) -> None:
    await message.reply("\\style X - изменить степень стилизации. "
                        "Чем больше X, тем больше внимания будет уделено стилю "
                        "и меньше содержанию. X - от 0 до 100. Начальное значение X - 75.",
                        reply_markup=keyboard_markup)


@dp.message_handler(commands=['style'])
async def style_cmd_handler(message: types.Message) -> None:
    new_alpha = int(message.text.split()[-1])
    if new_alpha < 0 or new_alpha > 100:
        await message.reply("Не могу применить данное значение стиля. "
                            "Оно должно быть от 0 до 100.", reply_markup=keyboard_markup)
    else:
        old_alpha = int(model.alpha * 100)
        model.alpha = new_alpha / 100
        await message.reply(f"Изменяю значение стиля с {old_alpha} на {new_alpha}.", reply_markup=keyboard_markup)


@dp.message_handler(text='Загрузить содержание')
async def all_msg_handler(message: types.Message):
    await message.reply('Хорошо! Отправьте мне картинку с содержанием.')
    global TYPE
    TYPE = 'content'


@dp.message_handler(text='Загрузить стиль')
async def all_msg_handler(message: types.Message):
    await message.reply('Хорошо! Отправьте мне картинку со стилем.')
    global TYPE
    TYPE = 'style'


@dp.message_handler(text='Перенести стиль')
async def all_msg_handler(message: types.Message):
    if not os.path.isfile(f'{message.from_user.id}/content.jpg'):
        await message.reply('Загрузите картинку содержания', reply_markup=keyboard_markup)
        return
    elif not os.path.isfile(f'{message.from_user.id}/style.jpg'):
        await message.reply('Загрузите картинку стиля', reply_markup=keyboard_markup)
        return
    await message.reply('Переношу стиль...', reply_markup=keyboard_markup)
    await types.ChatActions.upload_photo()
    img_content = load_img(f'{message.from_user.id}/content.jpg').to('cuda')
    img_style = load_img(f'{message.from_user.id}/style.jpg', img_size=768).to('cuda')
    result = model(img_content.unsqueeze(0), img_style.unsqueeze(0))
    save_image(result, f'{message.from_user.id}/result.jpg')
    await bot.send_photo(message.from_user.id, open(f'{message.from_user.id}/result.jpg', 'rb'))


@dp.message_handler(content_types=['photo'])
async def handle_photo(message):
    if not os.path.isdir(f'./{message.from_user.id}'):
        await aio_os.mkdir(f'./{message.from_user.id}')
    await message.photo[-1].download(f'{message.from_user.id}/{TYPE}.jpg')
    if TYPE == 'content':
        await bot.send_message(message.chat.id, "Отлично, содержание загружено!", reply_markup=inline_keyboard_markup)
    if TYPE == 'style':
        await bot.send_message(message.chat.id, "Отлично, стиль загружен!")


@dp.callback_query_handler()
async def apply_cezanne_handler(callback_query: types.CallbackQuery):
    await callback_query.message.reply('Переношу стиль...', reply_markup=keyboard_markup)
    await types.ChatActions.upload_photo()

    chat_id = callback_query.message.chat.id
    image = Image.open(f'{chat_id}/content.jpg').convert('RGB')
    pre_process = transforms.Compose([transforms.Resize(256),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                      ])
    image = pre_process(image).unsqueeze(0).to('cuda')

    fake_image = model_gan(image)
    save_image(fake_image.detach(), f'{chat_id}/result.jpg', normalize=True)
    await bot.send_photo(chat_id, open(f'{chat_id}/result.jpg', 'rb'))


if __name__ == '__main__':
    executor.start_polling(dp)
