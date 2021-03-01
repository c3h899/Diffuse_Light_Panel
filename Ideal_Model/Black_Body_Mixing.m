%% Housekeeping
close all;
clear; clc;

%% Input Parameters
T0 = 3000; % Black Body Temperature in [k]
T1 = 4500; % Black Body Temperature in [k]
T2 = 6000; % Black Body Temperature in [k]

% Wavelength Resolution
lambda_nm = linspace(400, 700, 1000); % Wavelength in [nm]
lambda = lambda_nm.*(1e-9); % Wavelength in [m]

% Temperature Sweep
sweep_inc = 100; % Delta_T in Kelvin/Celcius
sweep_T = (1800:sweep_inc:6500); % Temperature Steps

% Plot T's
plot_T = [1800, 2400, 2700, 3000, 3400, 4000, 5000, 5600, 6000, 6500];

%% Main Loop
resp0 = planck_law(lambda, T0);
norm0 = trapz(resp0); resp0 = resp0./norm0;
resp1 = planck_law(lambda, T1);
norm1 = trapz(resp1); resp1 = resp1./norm1;
resp2 = planck_law(lambda, T2);
norm2 = trapz(resp2); resp2 = resp2./norm2;
num_order = 10^(-1*round(log10(max(resp1))));

figure('name', 'Spectral Response');
plot(lambda_nm, resp0.*num_order, 'Color', (0.5).*[1 0 0]); hold on;
plot(lambda_nm, resp1.*num_order, 'Color', (0.5).*[0 1 0]);
plot(lambda_nm, resp2.*num_order, 'Color', (0.5).*[0 0 1]);
title('Tri-Color Approximation of Black Body');
xlabel('Wavelength / nm'); ylabel('Spectral Radiance / abu');
s1 = subplot(4, 2, 1);
s2 = subplot(4, 2, [3 5]);
s3 = subplot(4, 2, 7);
grid on; 

Mix = zeros(length(sweep_T), 4); ii = 1; jj = 1;
for temp = sweep_T
	respT = planck_law(lambda, temp);
	normT = trapz(respT); respT = respT./normT;
	
	if(temp <= T1)
		test = @(x) 1 - fitness(respT, lin_mix2(x, resp0, resp1));
		x1 = (temp + T1)/(T0 + T1); x2 = 1 - x1;
		opt = [fminsearch(test, [x1 x2]) -inf];
	else
		test = @(x) 1 - fitness(respT, lin_mix2(x, resp1, resp2));
		x1 = (temp + T2)/(T1 + T2); x2 = 1 - x1;
		opt = [-inf fminsearch(test, [x1 x2])];
	end
	res = lin_mix3(opt, resp0, resp1, resp2);
	fit = fitness(respT, res);
	Mix(ii, :) = [opt, fit];
	
	% Pictures
	if(plot_T(jj) == temp)
		if(temp < T0)
			subplot(4, 2, 1);
			plot(lambda_nm, respT.*num_order, '.', 'Color', 0.3*[1 0 0]); hold on;
			plot(lambda_nm, res.*num_order, '-r'); % Below Range
			title('Selected Intensity Responses (abu)');
			%ylabel('T < T_{min}');
			%legend(sprintf('%i K', T0));
			set(gca,'xticklabel',[]);
		elseif(temp > T2)
			subplot(4, 2, 7);
			plot(lambda_nm, respT.*num_order, '.', 'Color', 0.3*[0 0 1]); hold on;
			plot(lambda_nm, res.*num_order, '-b'); % Above Range
			xlabel('Wavelength / nm');
			%ylabel('T_{max} > T');
			%legend(sprintf('%i K', T1));
		else
			subplot(4, 2, [3 5]);
			plot(lambda_nm, respT.*num_order, '.', 'Color', 0.3*[0 1 0]); hold on;
			plot(lambda_nm, res.*num_order, '-g'); % Between Range
			ylabel('Intensity / abu');
			%ylabel('T_{min} <= T <= T_{max}');
			%legend(sprintf('%i K', T2));
			set(gca,'xticklabel',[]);
		end
		jj = jj + 1;
	end
%	disp(fit);
	ii = ii +1;
end
linkaxes([s1, s2, s3], 'xy');
xlim([400, 700]); ylim([0 2]);

% Worst Case
valid_ival = find((sweep_T >= T0) & (sweep_T <= T2));
min_ii = find(Mix(:,4) == min(Mix(valid_ival,4)));
temp_wc = sweep_T(min_ii);

resp_wc = planck_law(lambda, temp_wc);
norm_wc = trapz(resp_wc); resp_wc = resp_wc./norm_wc;
res_wc = lin_mix3(Mix(min_ii, 1:3), resp0, resp1, resp2);

%figure('name', 'Worst Case Fit');
subplot(4, 2, [2,4,6,8]);
plot(lambda_nm, resp_wc.*num_order, '-r'); hold on;
plot(lambda_nm, res_wc.*num_order,  'Color', 0.3*[1 0 0]);
hold off;

title(sprintf('(Worst Case) R^2=%0.2f, %i K', Mix(min_ii, 4), sweep_T(min_ii)));
xlabel('Wavelength / nm'); ylabel('Intensity / abu');
legend('Ideal','Approx.'); grid on; ylim([0, 2]);
fmt.figure();

%% Phase Diagram
figure('name', 'Mixing Coefficient');
s4 = subplot(2,1,1);
plot(sweep_T, soft_var(Mix(:,1)), '-or'); hold on;
plot(sweep_T, soft_var(Mix(:,2)), '-og');
plot(sweep_T, soft_var(Mix(:,3)), '-ob'); hold off;
ylabel('Mixing Coefficient / abu');
set(gca,'xticklabel',[]);

s5 = subplot(2,1,2);
plot(sweep_T, Mix(:,4), '-ok');
xlabel('Temperature / K');
ylabel('R-Squared Correlation');

linkaxes([s4, s5], 'x');
xlim([T0, T2]);
fmt.figure();

%% Supporting Function
function [fit] = fitness(obs, est)
	% Goodness of fit
	% R2
	obs_est = (1/length(obs)).*sum(obs);
	var_y = obs - obs_est;
	ss_tot = sum(var_y.*var_y);
	resid = obs - est;
	ss_res = sum(resid.*resid);
	fit = 1 - ss_res/ss_tot;
	% RMS Error
%	fit = sqrt(1./length(obs).*sum((obs - est).^2));
end
function [Bv] = lin_mix2(x, resp0, resp1)
	y = soft_var(x);
	Bv = y(1).*resp0 + y(2).*resp1;
end
function [Bv] = lin_mix3(x, resp0, resp1, resp2)
	y = soft_var(x);
	Bv = y(1).*resp0 + y(2).*resp1 + y(3).*resp2;
end
function [Bv] = planck_law(lambda, temp)
	% Wavelength [Vector] (lambda) in m
	% Temperature [Constant] (temp) in Kelvin
	HC = const.H*const.c0;
	k1 = 2*HC.*const.c0;
	k2 = HC/(const.kB*temp);
	lambda2 = lambda.*lambda;
	lambda4 = lambda2.*lambda2;
	Bv = k1./((lambda4.*lambda).*(exp(k2./lambda) - 1)); % W / (sr m^3)
end
function [lin] = soft_var(x)
	lin = 2./(1 + exp(-x));
end
