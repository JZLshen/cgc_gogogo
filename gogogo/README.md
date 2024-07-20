# README

# 环境与第三方库

1.  openmp
    
*   需要用到OpenMP库。安装执行以下命令即可:

    sudo apt-get install libomp-dev

# 编译并运行

    # 编译
    cd /cgc_gogogo/gogogo
    make
    # 运行
    ./gogogo.exe 64 16 8 graph/1024_example_graph.txt embedding/1024.bin weight/W_64_16.bin weight/W_16_8.bin