FROM node:18-alpine
WORKDIR /app
ENV NEXT_PUBLIC_API_URL=http://localhost:8000
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
