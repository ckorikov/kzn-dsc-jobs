FROM python:3.6.7

RUN pip install pipenv

WORKDIR /app

ADD Pipfile ./
ADD Pipfile.lock ./
RUN set -ex && pipenv install --deploy --system

ADD telegram-bot /app/telegram-bot

ENV key None
ENTRYPOINT ["/bin/bash", "-c", "python telegram-bot/bot.py $key"]
