FROM nekodigi/gpu_essentials

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

#RUN python -c "from opencv_fixer import AutoFix; AutoFix()"
