randn('state',0);
n = 30;                                             % number of stocks
B = 5;                                              % budget
Beta = 0.1;                                         % fixed cost
Alpha = 0.05*rand(n,1);                             % linear cost
mu = linspace(0.03,0.3,30)';                        % mean return
stddev = linspace(0,0.4,30);
t = randn(n);
s = t*t';
Sigma = diag(diag(s))^(-.5)*s*s'*diag(diag(s))^(-.5);
Sigma = diag(stddev)*Sigma*diag(stddev);            % covariance of return
Rmin = 0.4;                                         % minimum return

