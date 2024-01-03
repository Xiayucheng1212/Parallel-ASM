make test

./test ./testcase/t1 8 4 
./test ./testcase/t2 16 4
./test ./testcase/t3 17 4
./test ./testcase/t4 1023 128
./test ./testcase/t5 1024 16
./test ./testcase/t6 2048 64
./test ./testcase/t7 4096 64 # -> error testcase with P_size >= 128
./test ./testcase/t8 4096 128
./test ./testcase/t9 10000 256
./test ./testcase/t10 100000 1024

make asm_cpu_seq asm_cpu_omp asm_gpu asm_blocked
echo "judging openmp cpu version"
for i in {1..10}
do
    ./asm_cpu_seq ./testcase/t${i}_T ./testcase/t${i}_P ./testcase/t${i}.ans
    ./asm_cpu_omp ./testcase/t${i}_T ./testcase/t${i}_P ./testcase/t${i}.out
    diff ./testcase/t${i}.out ./testcase/t${i}.ans
done

echo "judging normal gpu version"
for i in {1..4}
do
    srun --gres=gpu:1 --nodes=1 ./asm_gpu ./testcase/t${i}_T ./testcase/t${i}_P ./testcase/t${i}.out
    diff ./testcase/t${i}.out ./testcase/t${i}.ans
done

echo "judging blocked gpu version"
for i in {1..10}
do
    srun --gres=gpu:1 --nodes=1 ./asm_blocked ./testcase/t${i}_T ./testcase/t${i}_P ./testcase/t${i}.out
    diff ./testcase/t${i}.out ./testcase/t${i}.ans
done