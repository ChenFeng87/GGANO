clear all;clc;
load('Emp_cov.mat')
din_N=10; % Dimension
time_T=20; % Time
rho=0.05; % \rho
sample_N=50000; % The sample size
yita_1 = rho/sample_N; 
yita_i = 3*rho/(2*sample_N);
yita_T = rho/sample_N;
% lambda controls sparsity, beta controls similarity between adjacent \Theta
lambda = 0.1;
beta =100;
% h_i is the time interval between adjacent timestamps
h_i=1;
Z_coeff=rho*h_i/(rho*h_i+2*beta);

%% Initialize
Theta0 = cell(time_T,1); % 1,2,...,T
Y_0 = cell(time_T,1); % 1,2,...,T
U_0 = cell(time_T,1); % 1,2,...,T
for i=1:time_T
    Theta0{i,1}=eye(din_N)*0.1;
    Y_0{i,1}=eye(din_N)*0.1;
    U_0{i,1}=eye(din_N)*0.1;
end
Z_0 = cell(time_T-1,1); % 2,3,...,T
V_0 = cell(time_T-1,1); % 2,3,...,T
for i=1:time_T-1
    Z_0{i,1}=eye(din_N)*0.1;
    V_0{i,1}=eye(din_N)*0.1;
end

%% updata
for step=1:1000 
    for i=1:time_T
        Y_old0{i,1}=Y_0{i,1};
    end
    for i=1:time_T-1
        Z_old0{i,1}=Z_0{i,1};
    end
    
   % theta_updata 
    for i=1:time_T
        if i==1
            A_tempor = (Y_0{i,1}-U_0{i,1} + Theta0{i+1,1}-Z_0{i,1}+V_0{i,1})/2;
            B_deco = yita_1*(A_tempor+A_tempor.')-S_i{i,1};
            [Q,D] = eig(B_deco);
            Theta0{i,1} = 1/(4*yita_1)*Q*(D+sqrt(D*D+8*yita_1*eye(din_N)))*Q.';
        elseif i > 1 && i < time_T
            A_tempor = (Y_0{i,1}-U_0{i,1}+Theta0{i-1,1}+Z_0{i-1,1}-V_0{i-1,1}+Theta0{i+1,1}-Z_0{i,1}+V_0{i,1})/3;
            B_deco = yita_i*(A_tempor+A_tempor.')-S_i{i,1};
            [Q,D] = eig(B_deco);
            Theta0{i,1} = 1/(4*yita_i)*Q*(D+sqrt(D*D+8*yita_i*eye(din_N)))*Q.';
        else
            A_tempor = (Y_0{i,1}-U_0{i,1}+Theta0{i-1,1}+Z_0{i-1,1}-V_0{i-1,1})/2;
            B_deco = yita_T*(A_tempor+A_tempor.')-S_i{i,1};
            [Q,D] = eig(B_deco);
            Theta0{i,1} = 1/(4*yita_T)*Q*(D+sqrt(D*D+8*yita_T*eye(din_N)))*Q.';
        end
    end

    % Y_updata
    for os=1:time_T
        for i=1:din_N
            for j=1:din_N
                if i~=j
                    aij_tempor = Theta0{os,1}(i,j)+U_0{os,1}(i,j);
                    if abs(aij_tempor)>lambda/rho
                        Y_0{os,1}(i,j) = sign(aij_tempor)*(abs(aij_tempor)-lambda/rho);
                    else
                        Y_0{os,1}(i,j) =0;
                    end
                end
            end
        end
    end
    
    % Z updata
    for os=2:time_T
        Z_0{os-1,1} = Z_coeff*(Theta0{os,1}-Theta0{os-1,1}+V_0{os-1,1});
    end

    % U updata
    for os=1:time_T
        U_0{os,1} = U_0{os,1}+ rho*(Theta0{os,1}-Y_0{os,1});
    end
    % V updata
    for os=2:time_T
        V_0{os-1,1} = V_0{os-1,1}+rho*(Theta0{os,1}-Theta0{os-1,1}-Z_0{os-1,1});
    end
    
    % computer residual values
    r_k = 0;
    for os = 1:time_T
        r_k = r_k + norm(Theta0{os,1}-diag(diag(Theta0{os,1}))-Y_0{os,1}+diag(diag(Y_0{os,1})),'fro');
    end
    fprintf('########## primal residuals ##########\n');
    r_k/10
    
    s_k0 =0;
    for os=1:time_T
        s_k0 = s_k0 + norm(Y_0{os,1}-Y_old0{os,1},'fro');
    end
    for os=1:time_T-1
        s_k0 = s_k0 + norm(Z_0{os,1}-Z_old0{os,1},'fro');
    end
    fprintf('########## dual residuals ##########\n');
    s_k0=s_k0 * rho
end

save('prediction_network.mat','Theta0')