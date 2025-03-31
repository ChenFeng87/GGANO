clear all
clc
filename = 'Data.mat';   
data = load(filename);
value = data.array;  
matrix_1 = zscore(value);
rng(2023);  
S_i = cell(20,1)
for i=1:20
    S_i{i,1}=cov(matrix_1(randperm(size(matrix_1, 1), 50000), :))*49999/50000;
end
save('Emp_cov.mat','S_i') 




