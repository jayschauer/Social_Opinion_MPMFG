% plot the mean field
[X,T] = meshgrid(0:dx:X_max, 0:dt:T_max);
surf(X, T, squeeze(base_m(1, :, :))); hold on
surf(X, T, squeeze(base_m(2, :, :)));