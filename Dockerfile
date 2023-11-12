FROM python:3.10.8

# Set the working directory in the container
WORKDIR /app

# Copy the local code to the container
COPY . .

# Install FastAPI and Uvicorn
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip cache purge

# Expose the port the app runs on
EXPOSE 80

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
