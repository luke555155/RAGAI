FROM node:18-alpine
WORKDIR /app
# Use 127.0.0.1 to avoid systems resolving 'localhost' to IPv6 (::1)
# which would fail because the backend only listens on IPv4
ENV NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
