FROM nginx:1.25

# Remove the default nginx configuration file
RUN rm /etc/nginx/conf.d/default.conf

# Copy the custom nginx configuration file to the appropriate location
COPY ./chatbot_app/deploy/nginx/nginx.conf /etc/nginx/conf.d
