# Step 1: Use a specific Python version on a slim Linux base.
# The --platform flag ensures it builds for amd64 architecture.
FROM --platform=linux/amd64 python:3.11-slim

# Step 2: Set a working directory inside the container.
# This is where your code will live and run.
WORKDIR /app

# Step 3: Copy only the requirements file first.
# This leverages Docker's layer caching. It will only re-run pip install
# if the requirements.txt file changes, speeding up future builds.
COPY requirements.txt .

# Step 4: Install Python dependencies from requirements.txt.
# The --no-cache-dir flag keeps the image size smaller.
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy your entire project into the working directory.
# This includes your run.py, app.py, and all "Collection X" folders.
COPY . .

# Step 6: Define the command to run your application.
# We use "python -u" to ensure that print statements are sent directly
# to the terminal without being buffered, so you can see the output in real-time.
CMD ["python", "-u", "run.py"]