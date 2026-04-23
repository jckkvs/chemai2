FROM node:20-alpine

WORKDIR /app

# package.jsonをコピー
COPY package*.json ./

# 依存関係のインストール
RUN npm install

# ソースコードをコピー
COPY . .

EXPOSE 3000

CMD ["npm", "run", "dev"]
