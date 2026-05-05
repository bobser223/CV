FROM ubuntu:24.04

RUN apt-get update && apt-get install -y \
    g++ \
    cmake \
    ninja-build \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp

RUN git clone https://github.com/yhirose/cpp-httplib.git && \
    cd cpp-httplib && \
    cmake -S . -B build -DHTTPLIB_COMPILE=OFF && \
    cmake --install build

WORKDIR /app

COPY . .

RUN cmake -S . -B build && cmake --build build

CMD ["./build/worm_server"]