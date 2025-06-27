FROM node:18-alpine
WORKDIR /app
# Use the backend service name so the frontend container can reach it
ENV NEXT_PUBLIC_API_URL=http://backend:8000
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
