close all;
clear;
clc;

% Calculation domain convention:
%   1 = solid matrix
%   0 = fluid pore

%% Output settings
outputRootBase = 'output';
runName = ['LBM_Run_', char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'))];
outputRoot = fullfile(outputRootBase, runName);
outputDirs = init_output_dirs(outputRoot);
disp(['LBM output root: ', outputRoot]);

%% QSGS porous medium
ad = 1;       % 1 = generate QSGS porous medium
lx = 100;     % size in flow direction
ly = 100;     % size in transverse direction
n = 0.6;      % target porosity
lb_ = 1;      % 1 = median filter smoothing
SIZE_ = 3;    % median filter size
FF = 2;       % median filter iterations
cdd = 0.01;   % seed probability

d1_grow = 0.01; d2_grow = 0.01; d3_grow = 0.01; d4_grow = 0.01;
d7_grow = 0.03; d8_grow = 0.03; d9_grow = 0.03; d10_grow = 0.05;

rngState = rng;

if ad == 1
    for zhedie = 1:1
        numtotal_need = (1 - n) * lx * ly;
        numsolid = 0;
        arrgrid = zeros(lx, ly);
        solid = zeros(lx * ly, 2);

        indices = randperm(lx * ly);
        for index = 1:(lx * ly)
            [i, j] = ind2sub([lx, ly], indices(index));
            if rand() < cdd
                numsolid = numsolid + 1;
                arrgrid(i, j) = 1;
                solid(numsolid, 1) = i;
                solid(numsolid, 2) = j;
            end
        end

        disp(['Seed count: ', num2str(numsolid)]);
        if numsolid == 0
            error('QSGS generated no seed. Increase cdd or rerun.');
        end

        tnumsolid = numsolid;
        while tnumsolid <= numtotal_need
            old_numsolid = numsolid;

            for index_solid = 1:tnumsolid
                index_i = solid(index_solid, 1);
                index_j = solid(index_solid, 2);

                if tnumsolid >= numtotal_need; break; end
                if index_i < lx
                    i = index_i + 1; j = index_j;
                    if arrgrid(i, j) == 0 && rand() < d1_grow
                        numsolid = numsolid + 1; arrgrid(i, j) = 1;
                        solid(numsolid, 1) = i; solid(numsolid, 2) = j;
                        tnumsolid = numsolid;
                    end
                end

                if tnumsolid >= numtotal_need; break; end
                if index_j < ly
                    i = index_i; j = index_j + 1;
                    if arrgrid(i, j) == 0 && rand() < d2_grow
                        numsolid = numsolid + 1; arrgrid(i, j) = 1;
                        solid(numsolid, 1) = i; solid(numsolid, 2) = j;
                        tnumsolid = numsolid;
                    end
                end

                if tnumsolid >= numtotal_need; break; end
                if index_i > 1
                    i = index_i - 1; j = index_j;
                    if arrgrid(i, j) == 0 && rand() < d3_grow
                        numsolid = numsolid + 1; arrgrid(i, j) = 1;
                        solid(numsolid, 1) = i; solid(numsolid, 2) = j;
                        tnumsolid = numsolid;
                    end
                end

                if tnumsolid >= numtotal_need; break; end
                if index_j > 1
                    i = index_i; j = index_j - 1;
                    if arrgrid(i, j) == 0 && rand() < d4_grow
                        numsolid = numsolid + 1; arrgrid(i, j) = 1;
                        solid(numsolid, 1) = i; solid(numsolid, 2) = j;
                        tnumsolid = numsolid;
                    end
                end

                if tnumsolid >= numtotal_need; break; end
                if index_i < lx && index_j < ly
                    i = index_i + 1; j = index_j + 1;
                    if arrgrid(i, j) == 0 && rand() < d7_grow
                        numsolid = numsolid + 1; arrgrid(i, j) = 1;
                        solid(numsolid, 1) = i; solid(numsolid, 2) = j;
                        tnumsolid = numsolid;
                    end
                end

                if tnumsolid >= numtotal_need; break; end
                if index_i > 1 && index_j < ly
                    i = index_i - 1; j = index_j + 1;
                    if arrgrid(i, j) == 0 && rand() < d8_grow
                        numsolid = numsolid + 1; arrgrid(i, j) = 1;
                        solid(numsolid, 1) = i; solid(numsolid, 2) = j;
                        tnumsolid = numsolid;
                    end
                end

                if tnumsolid >= numtotal_need; break; end
                if index_i > 1 && index_j > 1
                    i = index_i - 1; j = index_j - 1;
                    if arrgrid(i, j) == 0 && rand() < d9_grow
                        numsolid = numsolid + 1; arrgrid(i, j) = 1;
                        solid(numsolid, 1) = i; solid(numsolid, 2) = j;
                        tnumsolid = numsolid;
                    end
                end

                if tnumsolid >= numtotal_need; break; end
                if index_i < lx && index_j > 1
                    i = index_i + 1; j = index_j - 1;
                    if arrgrid(i, j) == 0 && rand() < d10_grow
                        numsolid = numsolid + 1; arrgrid(i, j) = 1;
                        solid(numsolid, 1) = i; solid(numsolid, 2) = j;
                        tnumsolid = numsolid;
                    end
                end
            end

            if tnumsolid >= numtotal_need
                break;
            end

            if numsolid == old_numsolid
                warning('QSGS growth stalled before reaching the target porosity.');
                break;
            end
        end

        if lb_ == 1
            for asdds = 1:FF
                arrgrid = medfilt2(arrgrid, [SIZE_, SIZE_]);
            end
            aa = sum(arrgrid(:) == 0);
            disp(['Porosity after smoothing: ', num2str(aa / lx / ly)]);
        end
    end
    disp('QSGS generation finished.');
end

if ~exist('arrgrid', 'var')
    error('No pore structure arrgrid was generated.');
end

%% Poiseuille validation switch
Poiseuille = 0;
if Poiseuille == 1
    rows = 100;
    cols = 101;
    arrgrid = zeros(rows, cols);
    arrgrid(:, 1) = 1;
    arrgrid(:, cols) = 1;
end

%% LBM initialization
LC = arrgrid;
Nx = size(LC, 1);
Ny = size(LC, 2);

if Poiseuille ~= 1
    Nx = Nx + 10;
    LC_ = zeros(Nx, Ny);
    LC_(6:end-5, :) = LC;
    LC = LC_;
    coreRows = 6:(Nx - 5);
else
    coreRows = 1:Nx;
end

rho = ones(Nx, Ny);

% D2Q9 model
wi = [1/9 1/9 1/9 1/9 1/36 1/36 1/36 1/36 4/9];
cx = [1 0 -1 0 1 -1 -1 1 0];
cy = [0 1 0 -1 1 1 -1 -1 0];
opp = [3, 4, 1, 2, 7, 8, 5, 6, 9];

rhoin = 1.00001;
rhoout = 1;
tao = 1;
v_ = 1/3 * (tao - 0.5);
disp(['Kinematic viscosity = ', num2str(v_)]);

t_savect = 500;
vtk_savect = 10000;
velocityResidualCriterion = 1e-4;
massBalanceCriterion = 1e-3;
permeabilityChangeCriterion = 1e-4;
time_total = 50000;
ht_ = 0;

rho(1, :) = rhoin;
rho(Nx, :) = rhoout;
f = zeros(Nx, Ny, 9);
u = zeros(Nx, Ny);
v = zeros(Nx, Ny);

for i = 1:Nx
    for j = 1:Ny
        for l = 1:9
            f(i, j, l) = rho(i, j) * wi(l);
        end
    end
end

obst2 = LC;
boundary_ = find(obst2 == 1);
solid = obst2 == 1;

samplePorosity = nnz(arrgrid == 0) / numel(arrgrid);
domainPorosity = nnz(LC == 0) / numel(LC);
porosity = samplePorosity;
disp(['Sample porosity = ', num2str(samplePorosity)]);
disp(['LBM domain porosity including buffer = ', num2str(domainPorosity)]);

current_u = zeros(Nx, Ny);
current_v = zeros(Nx, Ny);

%% Poiseuille analytical velocity profile
if Poiseuille == 1
    uy_ = zeros(size(LC));
    uy_ = uy_(:, 1:cols);
    rho_ = rho(:, 1:cols);
    [lxx, lyy] = size(uy_);
    for i = 1:lxx
        for j = 1:cols
            jy = j - 1;
            miu = rho_(i, j) * v_;
            bbb = (rhoin - rhoout) / 3 / lxx * jy * (jy - (lyy - 1));
            uy_(i, j) = -bbb / (2 * miu);
        end
    end
end

%% Main LBM loop
tic;
U = zeros(Nx, Ny);
converged = false;
finalReady = false;
finalTime = 0;
d = NaN;
K = NaN;
previousK = NaN;
permeabilityChange = Inf;
TI = 0;
history = zeros(ceil(time_total / t_savect), 8);
historyCount = 0;

for time = 1:time_total
    f0 = collecsion(f, tao, u, v, cx, cy, rho, wi, Nx, Ny, obst2);
    f = straming(f0, cx, cy);
    f = boundary(f, obst2, opp, Nx, rhoin, rhoout);
    [u, v, rho, f] = calculation(f, cx, cy, Nx, Ny, wi, rhoin, rhoout);
    u(solid) = 0;
    v(solid) = 0;
    rho(solid) = 1;

    if mod(time, t_savect) == 0
        u__ = u;
        v__ = v;
        u__(boundary_) = 0;
        v__(boundary_) = 0;

        Rho_ = rho;
        Rho_(boundary_) = 1;

        TI = toc;
        U = sqrt(u__.^2 + v__.^2);

        U_t = U(coreRows, :);
        Rho_t = Rho_(coreRows, :);
        u_t = u__(coreRows, :);
        v_t = v__(coreRows, :);

        if ht_ == 1
            figure(2);
            imagesc(U);
            axis equal tight;
            colormap jet;
            colorbar;
            title(['Step: ', num2str(time)]);
            xlabel('Y index');
            ylabel('X index');
            drawnow;
        end

        if Poiseuille == 1
            figure(3);
            plot((1:cols) - 1, uy_(floor(rows / 2), :), 'b--', ...
                (1:cols) - 1, u__(floor(rows / 2), :), 'r--o');
            xlabel('Section width');
            ylabel('Velocity');
            title('Poiseuille validation');
            legend('Analytical', 'LBM', 'Location', 'NorthEast');
        end

        d = compute_residual(u__, v__, current_u, current_v);

        permNow = compute_darcy_permeability(u__, Rho_, obst2, coreRows, v_);
        K = permNow.permeability_lattice;

        % Relative permeability change between two consecutive checks.
        if isfinite(previousK) && isfinite(K)
            permeabilityScale = max([abs(previousK), abs(K), eps]);
            permeabilityChange = abs(K - previousK) / permeabilityScale;
        else
            permeabilityChange = Inf;
        end

        historyCount = historyCount + 1;
        history(historyCount, :) = [time, K, permNow.delta_pressure_sample, ...
            permNow.darcy_velocity_mean, d, permNow.mass_balance_error, ...
            permeabilityChange, TI];
        disp(['Step ', num2str(time), ...
            '  permeability = ', num2str(K), ...
            '  delta_p_sample = ', num2str(permNow.delta_pressure_sample), ...
            '  q_mean = ', num2str(permNow.darcy_velocity_mean), ...
            '  velocity_residual = ', num2str(d), ...
            '  mass_balance_error = ', num2str(permNow.mass_balance_error), ...
            '  permeability_change = ', num2str(permeabilityChange), ...
            '  elapsed = ', num2str(TI), ' s']);

        finalReady = true;
        finalTime = time;

        %% Periodic snapshot files
        if mod(time, vtk_savect) == 0
            snapshotMetrics = build_snapshot_metrics(time, permNow, d, ...
                permeabilityChange, TI, porosity, rhoin, rhoout, v_);
            pltwrite(u__, v__, Rho_, U, obst2, time, outputDirs.tecplot, ['time_', num2str(time)]);
            vtk(u__, v__, Rho_, Nx, Ny, U, time, obst2, outputDirs.vtk, ['time_', num2str(time)]);
            save_snapshot_mat(outputDirs.data_mat, time, u__, v__, Rho_, U, obst2, arrgrid, snapshotMetrics);
        end

        %% Convergence check
        velocityConverged = isfinite(d) && d < velocityResidualCriterion;
        massBalanceConverged = isfinite(permNow.mass_balance_error) && ...
            permNow.mass_balance_error < massBalanceCriterion;
        permeabilityConverged = isfinite(permeabilityChange) && ...
            permeabilityChange < permeabilityChangeCriterion;

        if velocityConverged && massBalanceConverged && permeabilityConverged
            converged = true;
            disp(['All convergence criteria reached at step ', num2str(time), ...
                ': velocity residual < ', num2str(velocityResidualCriterion), ...
                ', mass balance error < ', num2str(massBalanceCriterion), ...
                ', permeability change < ', num2str(permeabilityChangeCriterion), '.']);
            break;
        else
            current_u = u__;
            current_v = v__;
            previousK = K;
        end
    end
end

history = history(1:historyCount, :);

if ~finalReady
    finalTime = time;
    TI = toc;
    u__ = u;
    v__ = v;
    u__(boundary_) = 0;
    v__(boundary_) = 0;
    Rho_ = rho;
    Rho_(boundary_) = 1;
    U = sqrt(u__.^2 + v__.^2);
end

fieldsFull = derive_fields(u__, v__, Rho_, U, obst2);
fieldsCore = crop_fields(fieldsFull, coreRows);
maskCore = obst2(coreRows, :);

metrics = calculate_metrics(u__, Rho_, U, obst2, arrgrid, coreRows, ...
    rhoin, rhoout, v_, tao, finalTime, TI, d, permeabilityChange, ...
    converged, samplePorosity, domainPorosity, rngState, ...
    velocityResidualCriterion, massBalanceCriterion, permeabilityChangeCriterion);

save_history_csv(outputDirs.metrics, history);
save_final_outputs(outputDirs, finalTime, fieldsCore, fieldsFull, maskCore, obst2, ...
    arrgrid, metrics, u__, v__, Rho_, U, Nx, Ny);

disp('Final post-processing finished.');
disp(['All files were saved under: ', outputRoot]);

%% ================================================================
%% Local functions
%% ================================================================

function dirs = init_output_dirs(outputRoot)
dirs.root = outputRoot;
dirs.tecplot = fullfile(outputRoot, '01_Tecplot_PLT');
dirs.vtk = fullfile(outputRoot, '02_ParaView_VTK');
dirs.data_mat = fullfile(outputRoot, '03_MAT_Data');
dirs.csv = fullfile(outputRoot, '04_CSV_Field_Matrices');
dirs.fig_png = fullfile(outputRoot, '05_MATLAB_CloudMaps_PNG');
dirs.fig_fig = fullfile(outputRoot, '06_MATLAB_Figures_FIG');
dirs.metrics = fullfile(outputRoot, '07_Metrics');

names = fieldnames(dirs);
for k = 1:numel(names)
    ensure_dir(dirs.(names{k}));
end
end

function ensure_dir(folder)
if ~exist(folder, 'dir')
    mkdir(folder);
end
end

function f0 = collecsion(f, tao, u, v, cx, cy, rho, wi, Nx, Ny, obst2)
fluid = obst2 == 0;

t1 = u.^2 + v.^2;
feq = zeros(Nx, Ny, 9);
f0 = f;

for l = 1:9
    cu = u .* cx(l) + v .* cy(l);
    feq(:, :, l) = wi(l) * rho .* ...
        (1 + 3 * cu + 4.5 * cu.^2 - 1.5 * t1);

    temp = f(:, :, l) - (f(:, :, l) - feq(:, :, l)) / tao;

    f0_l = f0(:, :, l);
    f0_l(fluid) = temp(fluid);
    f0(:, :, l) = f0_l;
end
end

function f = straming(f0, cx, cy)
f = zeros(size(f0));
for l = 1:9
    f(:, :, l) = circshift(f0(:, :, l), [cx(l), cy(l)]);
end
end

function f = boundary(f, obst2, opp, Nx, rhoin, rhoout) %#ok<INUSD>
tempt = zeros(size(f));
for i = 1:9
    tempt(:, :, i) = f(:, :, opp(i));
end
f = (1 - obst2) .* f + obst2 .* tempt;
end

function [u, v, rho, f] = calculation(f, cx, cy, Nx, Ny, wi, rhoin, rhoout)
rho = zeros(Nx, Ny);
u = zeros(Nx, Ny);
v = zeros(Nx, Ny);

for i = 1:Nx
    for j = 1:Ny
        rho(i, j) = sum(f(i, j, :));
        usum = 0;
        vsum = 0;
        for l = 1:9
            usum = f(i, j, l) * cx(l) + usum;
            vsum = f(i, j, l) * cy(l) + vsum;
        end
        u(i, j) = usum / rho(i, j);
        v(i, j) = vsum / rho(i, j);
    end
end

% Non-equilibrium extrapolation pressure boundary.
rho(1, :) = rhoin;
rho(Nx, :) = rhoout;
u(1, :) = u(2, :);
u(Nx, :) = u(Nx - 1, :);
v(1, :) = v(2, :);
v(Nx, :) = v(Nx - 1, :);

t1 = u.^2 + v.^2;
t2 = zeros(Nx, Ny, 9);
feq = zeros(Nx, Ny, 9);
for l = 1:9
    t2(:, :, l) = u .* cx(l) + v .* cy(l);
    feq(:, :, l) = wi(l) * rho .* (1 + 3 * t2(:, :, l) + 4.5 * t2(:, :, l).^2 - 1.5 * t1);
end
for k = 1:9
    f(1, :, k) = feq(1, :, k) + f(2, :, k) - feq(2, :, k);
    f(Nx, :, k) = feq(Nx, :, k) + f(Nx - 1, :, k) - feq(Nx - 1, :, k);
end
end

function d = compute_residual(u, v, current_u, current_v)
diff_num = sum((u(:) - current_u(:)).^2 + (v(:) - current_v(:)).^2);
diff_den = sum(u(:).^2 + v(:).^2);
d = sqrt(diff_num) / (sqrt(diff_den) + 1e-30);
end

function perm = compute_darcy_permeability(u, rho, maskFull, coreRows, nu)
% Darcy permeability in LBM lattice units.
%
% Flow direction is the first matrix dimension, i.e. row/X direction.
% For QSGS samples with buffer layers, pressure and flux are measured on
% the fluid rows immediately outside the sample faces. This avoids mixing
% the pressure loss of the artificial buffer into the sample permeability.
% If no buffer exists, the first and last core rows are used instead.

cs2 = 1 / 3;
pressure = rho * cs2;
nxFull = size(maskFull, 1);
H = size(maskFull, 2);
L = numel(coreRows);

if coreRows(1) > 1 && coreRows(end) < nxFull
    inletRow = coreRows(1) - 1;
    outletRow = coreRows(end) + 1;
else
    inletRow = coreRows(1);
    outletRow = coreRows(end);
end

inletFluid = maskFull(inletRow, :) == 0;
outletFluid = maskFull(outletRow, :) == 0;

if any(inletFluid)
    densityIn = mean(rho(inletRow, inletFluid));
    pressureIn = mean(pressure(inletRow, inletFluid));
    inletFlux = sum(u(inletRow, inletFluid));
else
    densityIn = NaN;
    pressureIn = NaN;
    inletFlux = 0;
end

if any(outletFluid)
    densityOut = mean(rho(outletRow, outletFluid));
    pressureOut = mean(pressure(outletRow, outletFluid));
    outletFlux = sum(u(outletRow, outletFluid));
else
    densityOut = NaN;
    pressureOut = NaN;
    outletFlux = 0;
end

deltaP = pressureIn - pressureOut;
darcyVelocityOutlet = outletFlux / H;
darcyVelocityInlet = inletFlux / H;
darcyVelocityMean = 0.5 * (darcyVelocityInlet + darcyVelocityOutlet);
referenceDensity = 0.5 * (densityIn + densityOut);
dynamicViscosity = referenceDensity * nu;

% At steady state, inlet and outlet Darcy velocities should agree. Their
% mean is used in Darcy's law to reduce single-plane discretization error.
% Darcy's law uses dynamic viscosity mu = rho_ref * nu. The absolute value
% keeps intrinsic permeability non-negative while signed fluxes and deltaP
% remain available separately for flow-direction diagnostics.
if isfinite(deltaP) && abs(deltaP) > eps && isfinite(dynamicViscosity)
    permeability = abs(dynamicViscosity * darcyVelocityMean * L / deltaP);
else
    permeability = NaN;
end

massBalanceError = abs(inletFlux - outletFlux) / max([abs(inletFlux), abs(outletFlux), eps]);

perm = struct();
perm.permeability_lattice = permeability;
perm.pressure_in_sample_face = pressureIn;
perm.pressure_out_sample_face = pressureOut;
perm.delta_pressure_sample = deltaP;
perm.inlet_face_row = inletRow;
perm.outlet_face_row = outletRow;
perm.inlet_flux_lattice = inletFlux;
perm.outlet_flux_lattice = outletFlux;
perm.mean_flux_lattice = 0.5 * (inletFlux + outletFlux);
perm.darcy_velocity_inlet = darcyVelocityInlet;
perm.darcy_velocity_outlet = darcyVelocityOutlet;
perm.darcy_velocity_mean = darcyVelocityMean;
perm.reference_density_lattice = referenceDensity;
perm.dynamic_viscosity_lattice = dynamicViscosity;
perm.mass_balance_error = massBalanceError;
perm.valid = double(isfinite(permeability));
end

function metrics = build_snapshot_metrics(time, perm, residual, permeabilityChange, ...
    elapsed_s, porosity, rhoin, rhoout, nu)
metrics = struct();
metrics.time_step = time;
metrics.permeability_lattice = perm.permeability_lattice;
metrics.pressure_in_sample_face = perm.pressure_in_sample_face;
metrics.pressure_out_sample_face = perm.pressure_out_sample_face;
metrics.delta_pressure_sample_lattice = perm.delta_pressure_sample;
metrics.inlet_flux_lattice = perm.inlet_flux_lattice;
metrics.outlet_flux_lattice = perm.outlet_flux_lattice;
metrics.mean_flux_lattice = perm.mean_flux_lattice;
metrics.darcy_velocity_inlet_lattice = perm.darcy_velocity_inlet;
metrics.darcy_velocity_outlet_lattice = perm.darcy_velocity_outlet;
metrics.darcy_velocity_mean_lattice = perm.darcy_velocity_mean;
metrics.reference_density_lattice = perm.reference_density_lattice;
metrics.dynamic_viscosity_lattice = perm.dynamic_viscosity_lattice;
metrics.mass_balance_error = perm.mass_balance_error;
metrics.velocity_residual = residual;
metrics.permeability_relative_change = permeabilityChange;
metrics.elapsed_s = elapsed_s;
metrics.porosity = porosity;
metrics.rhoin = rhoin;
metrics.rhoout = rhoout;
metrics.nu_lattice = nu;
end

function fields = derive_fields(u, v, rho, speed, mask)
fields = struct();
fluid = mask == 0;

fields.solid_mask = double(mask == 1);
fields.fluid_mask = double(fluid);
fields.density = rho;
fields.pressure = rho / 3.0;
fields.velocity_x = u;
fields.velocity_y = v;
fields.velocity_magnitude = speed;
fields.log10_velocity_magnitude = log10(speed + eps);

% MATLAB gradient returns the column-direction derivative first and the
% row-direction derivative second. In this model, columns are physical Y
% and rows are physical X (the main flow direction).
[dp_dy, dp_dx] = gradient(fields.pressure);
[du_dy, du_dx] = gradient(fields.velocity_x);
[dv_dy, dv_dx] = gradient(fields.velocity_y);

fields.pressure_gradient_x = dp_dx;
fields.pressure_gradient_y = dp_dy;
fields.pressure_gradient_magnitude = sqrt(dp_dx.^2 + dp_dy.^2);
fields.velocity_divergence = du_dx + dv_dy;
fields.vorticity = dv_dx - du_dy;
fields.shear_rate_magnitude = sqrt(2 * du_dx.^2 + 2 * dv_dy.^2 + (du_dy + dv_dx).^2);

fluidSpeed = speed(fluid);
if isempty(fluidSpeed)
    highThr = NaN;
    lowThr = NaN;
else
    highThr = percentile_value(fluidSpeed, 90);
    lowThr = percentile_value(fluidSpeed, 10);
end

fields.main_flow_channel_top10 = double(fluid & speed >= highThr);
fields.stagnant_zone_bottom10 = double(fluid & speed <= lowThr);
fields.local_flux_x = u;
fields.local_flux_y = v;
end

function value = percentile_value(x, pct)
x = x(:);
x = x(isfinite(x));
if isempty(x)
    value = NaN;
    return;
end
x = sort(x);
idx = round((pct / 100) * (numel(x) - 1)) + 1;
idx = max(1, min(numel(x), idx));
value = x(idx);
end

function cropped = crop_fields(fields, rows)
cropped = struct();
names = fieldnames(fields);
for k = 1:numel(names)
    data = fields.(names{k});
    if isnumeric(data) && ismatrix(data)
        cropped.(names{k}) = data(rows, :);
    else
        cropped.(names{k}) = data;
    end
end
end

function metrics = calculate_metrics(u, rho, speed, maskFull, maskCoreOriginal, coreRows, ...
    rhoin, rhoout, nu, tao, finalTime, elapsed_s, velocityResidual, ...
    permeabilityChange, converged, samplePorosity, domainPorosity, rngState, ...
    velocityResidualCriterion, massBalanceCriterion, permeabilityChangeCriterion)

rhoCore = rho(coreRows, :);
speedCore = speed(coreRows, :);
maskCore = maskFull(coreRows, :);
fluid = maskCore == 0;

perm = compute_darcy_permeability(u, rho, maskFull, coreRows, nu);
deltaRho = rhoin - rhoout;
L = size(maskCore, 1);
H = size(maskCore, 2);

if any(fluid(:))
    meanSpeedFluid = mean(speedCore(fluid));
    maxSpeedFluid = max(speedCore(fluid));
    meanPressureFluid = mean(rhoCore(fluid) / 3.0);
else
    meanSpeedFluid = NaN;
    maxSpeedFluid = NaN;
    meanPressureFluid = NaN;
end

metrics = struct();
metrics.final_time_step = finalTime;
metrics.converged = double(converged);
metrics.velocity_residual = velocityResidual;
metrics.velocity_residual_criterion = velocityResidualCriterion;
metrics.mass_balance_criterion = massBalanceCriterion;
metrics.permeability_relative_change = permeabilityChange;
metrics.permeability_change_criterion = permeabilityChangeCriterion;
metrics.elapsed_s = elapsed_s;
metrics.rhoin = rhoin;
metrics.rhoout = rhoout;
metrics.delta_rho = deltaRho;
metrics.pressure_in_sample_face = perm.pressure_in_sample_face;
metrics.pressure_out_sample_face = perm.pressure_out_sample_face;
metrics.delta_pressure_sample_lattice = perm.delta_pressure_sample;
metrics.tao = tao;
metrics.nu_lattice = nu;
metrics.sample_porosity = samplePorosity;
metrics.domain_porosity_including_buffer = domainPorosity;
metrics.core_length_lattice = L;
metrics.core_width_lattice = H;
metrics.inlet_face_row = perm.inlet_face_row;
metrics.outlet_face_row = perm.outlet_face_row;
metrics.inlet_flux_lattice = perm.inlet_flux_lattice;
metrics.outlet_flux_lattice = perm.outlet_flux_lattice;
metrics.mean_flux_lattice = perm.mean_flux_lattice;
metrics.darcy_velocity_inlet_lattice = perm.darcy_velocity_inlet;
metrics.darcy_velocity_outlet_lattice = perm.darcy_velocity_outlet;
metrics.darcy_velocity_mean_lattice = perm.darcy_velocity_mean;
metrics.reference_density_lattice = perm.reference_density_lattice;
metrics.dynamic_viscosity_lattice = perm.dynamic_viscosity_lattice;
metrics.permeability_lattice = perm.permeability_lattice;
metrics.permeability_valid = perm.valid;
metrics.mass_balance_error = perm.mass_balance_error;
metrics.velocity_criterion_met = double(isfinite(velocityResidual) && ...
    velocityResidual < velocityResidualCriterion);
metrics.mass_balance_criterion_met = double(isfinite(perm.mass_balance_error) && ...
    perm.mass_balance_error < massBalanceCriterion);
metrics.permeability_change_criterion_met = double(isfinite(permeabilityChange) && ...
    permeabilityChange < permeabilityChangeCriterion);
metrics.mean_speed_fluid = meanSpeedFluid;
metrics.max_speed_fluid = maxSpeedFluid;
metrics.mean_pressure_fluid = meanPressureFluid;
metrics.rng_seed_type = rngState.Type;
metrics.rng_seed_value = rngState.Seed;

% Keep the original core mask name explicit in the MAT file metrics context.
metrics.core_mask_rows = numel(coreRows);
metrics.original_mask_solid_fraction = nnz(maskCoreOriginal == 1) / numel(maskCoreOriginal);
end

function save_snapshot_mat(outdir, time, u, v, rho, speed, mask, arrgrid, metrics)
ensure_dir(outdir);
filename = fullfile(outdir, ['snapshot_time_', num2str(time), '.mat']);
save(filename, 'u', 'v', 'rho', 'speed', 'mask', 'arrgrid', 'metrics', '-v7.3');
disp(['MAT snapshot saved: ', filename]);
end

function save_history_csv(outdir, history)
ensure_dir(outdir);
filename = fullfile(outdir, 'convergence_history.csv');
fid = fopen(filename, 'w');
if fid == -1
    error('Cannot open convergence history file: %s', filename);
end
fprintf(fid, ['time_step,permeability_lattice,delta_pressure_sample_lattice,', ...
    'darcy_velocity_mean_lattice,velocity_residual,mass_balance_error,', ...
    'permeability_relative_change,elapsed_s\n']);
for i = 1:size(history, 1)
    fprintf(fid, '%d,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g\n', ...
        history(i, 1), history(i, 2), history(i, 3), history(i, 4), ...
        history(i, 5), history(i, 6), history(i, 7), history(i, 8));
end
fclose(fid);
disp(['Convergence history saved: ', filename]);
end

function save_final_outputs(outputDirs, finalTime, fieldsCore, fieldsFull, maskCore, maskFull, ...
    arrgrid, metrics, uFull, vFull, rhoFull, speedFull, Nx, Ny)

coreMatFile = fullfile(outputDirs.data_mat, ['final_core_fields_time_', num2str(finalTime), '.mat']);
fullMatFile = fullfile(outputDirs.data_mat, ['final_full_fields_time_', num2str(finalTime), '.mat']);
save(coreMatFile, 'fieldsCore', 'maskCore', 'arrgrid', 'metrics', '-v7.3');
save(fullMatFile, 'fieldsFull', 'maskFull', 'uFull', 'vFull', 'rhoFull', 'speedFull', 'metrics', '-v7.3');

save_fields_csv(fullfile(outputDirs.csv, 'core_domain'), fieldsCore);
save_fields_csv(fullfile(outputDirs.csv, 'full_domain'), fieldsFull);

save_metrics(outputDirs.metrics, metrics);

plot_all_cloud_maps(fieldsCore, maskCore, fullfile(outputDirs.fig_png, 'core_domain'), ...
    fullfile(outputDirs.fig_fig, 'core_domain'));

plot_vector_cloud(fieldsCore, maskCore, fullfile(outputDirs.fig_png, 'core_domain'), ...
    fullfile(outputDirs.fig_fig, 'core_domain'));

plot_streamline_cloud(fieldsCore, maskCore, fullfile(outputDirs.fig_png, 'core_domain'), ...
    fullfile(outputDirs.fig_fig, 'core_domain'));

pltwrite(fieldsCore.velocity_x, fieldsCore.velocity_y, fieldsCore.density, ...
    fieldsCore.velocity_magnitude, maskCore, finalTime, outputDirs.tecplot, 'final_core_converged');
pltwrite(uFull, vFull, rhoFull, speedFull, maskFull, finalTime, outputDirs.tecplot, 'final_full_converged');

vtk(fieldsCore.velocity_x, fieldsCore.velocity_y, fieldsCore.density, size(maskCore, 1), size(maskCore, 2), ...
    fieldsCore.velocity_magnitude, finalTime, maskCore, outputDirs.vtk, 'final_core_converged');
vtk(uFull, vFull, rhoFull, Nx, Ny, speedFull, finalTime, maskFull, outputDirs.vtk, 'final_full_converged');

disp(['Final core MAT saved: ', coreMatFile]);
disp(['Final full MAT saved: ', fullMatFile]);
end

function save_fields_csv(outdir, fields)
ensure_dir(outdir);
names = fieldnames(fields);
for k = 1:numel(names)
    data = fields.(names{k});
    if isnumeric(data) && ismatrix(data)
        filename = fullfile(outdir, [names{k}, '.csv']);
        writematrix(data, filename);
    end
end
end

function save_metrics(outdir, metrics)
ensure_dir(outdir);

csvFile = fullfile(outdir, 'final_metrics.csv');
txtFile = fullfile(outdir, 'final_metrics.txt');

write_metrics_file(csvFile, metrics, true);
write_metrics_file(txtFile, metrics, false);

disp(['Final metrics saved: ', csvFile]);
end

function write_metrics_file(filename, metrics, csvMode)
fid = fopen(filename, 'w');
if fid == -1
    error('Cannot open metrics file: %s', filename);
end

names = fieldnames(metrics);
if csvMode
    fprintf(fid, 'name,value\n');
end

for k = 1:numel(names)
    value = metrics.(names{k});
    if isnumeric(value)
        valueText = num2str(value, '%.15g');
    elseif ischar(value)
        valueText = value;
    else
        valueText = '<non_numeric>';
    end

    if csvMode
        fprintf(fid, '%s,%s\n', names{k}, valueText);
    else
        fprintf(fid, '%s = %s\n', names{k}, valueText);
    end
end
fclose(fid);
end

function plot_all_cloud_maps(fields, mask, pngDir, figDir)
ensure_dir(pngDir);
ensure_dir(figDir);

plotItems = {
    'solid_mask', 'Solid mask: 1 solid, 0 pore', 'gray', false;
    'fluid_mask', 'Fluid pore mask', 'gray', false;
    'density', 'Density field rho', 'jet', true;
    'pressure', 'Pressure field p = rho / 3', 'jet', true;
    'pressure_gradient_x', 'Pressure gradient, x direction', 'turbo', true;
    'pressure_gradient_y', 'Pressure gradient, y direction', 'turbo', true;
    'pressure_gradient_magnitude', 'Pressure gradient magnitude', 'turbo', true;
    'velocity_x', 'Velocity component ux', 'jet', true;
    'velocity_y', 'Velocity component uy', 'jet', true;
    'velocity_magnitude', 'Velocity magnitude |u|', 'turbo', true;
    'log10_velocity_magnitude', 'log10 velocity magnitude', 'turbo', true;
    'velocity_divergence', 'Velocity divergence', 'turbo', true;
    'vorticity', 'Vorticity', 'turbo', true;
    'shear_rate_magnitude', 'Shear-rate magnitude', 'turbo', true;
    'main_flow_channel_top10', 'Main flow channel: top 10% speed', 'hot', true;
    'stagnant_zone_bottom10', 'Stagnant zone: bottom 10% speed', 'cool', true;
    'local_flux_x', 'Local flux x component', 'jet', true;
    'local_flux_y', 'Local flux y component', 'jet', true;
    };

for k = 1:size(plotItems, 1)
    fieldName = plotItems{k, 1};
    if isfield(fields, fieldName)
        plot_scalar_cloud(fields.(fieldName), mask, fieldName, plotItems{k, 2}, ...
            plotItems{k, 3}, plotItems{k, 4}, pngDir, figDir);
    end
end
end

function plot_scalar_cloud(data, mask, fileStem, titleText, cmapName, hideSolid, pngDir, figDir)
fig = figure('Visible', 'off', 'Color', 'w');
plotData = data;

if hideSolid
    plotData(mask == 1) = NaN;
end

imagesc(plotData);
axis image tight;
set(gca, 'YDir', 'normal');
set(gca, 'FontName', 'Arial', 'FontSize', 11);
set(gca, 'Color', [0.15, 0.15, 0.15]);
xlabel('Y lattice index');
ylabel('X lattice index');
title(titleText, 'Interpreter', 'none');
apply_colormap(cmapName);
colorbar;

pngFile = fullfile(pngDir, [fileStem, '.png']);
figFile = fullfile(figDir, [fileStem, '.fig']);
saveas(fig, pngFile);
saveas(fig, figFile);
close(fig);
end

function apply_colormap(cmapName)
try
    colormap(cmapName);
catch
    colormap(jet);
end
end

function plot_vector_cloud(fields, mask, pngDir, figDir)
ensure_dir(pngDir);
ensure_dir(figDir);

u = fields.velocity_x;
v = fields.velocity_y;
speed = fields.velocity_magnitude;
u(mask == 1) = NaN;
v(mask == 1) = NaN;

[X, Y] = meshgrid(1:size(mask, 2), 1:size(mask, 1));
skip = max(1, round(min(size(mask)) / 35));

fig = figure('Visible', 'off', 'Color', 'w');
plotData = speed;
plotData(mask == 1) = NaN;
imagesc(plotData);
hold on;
axis image tight;
set(gca, 'YDir', 'normal');
set(gca, 'Color', [0.15, 0.15, 0.15]);
apply_colormap('turbo');
colorbar;

% In the displayed matrix, horizontal axis is column/Y and vertical axis is row/X.
quiver(X(1:skip:end, 1:skip:end), Y(1:skip:end, 1:skip:end), ...
    v(1:skip:end, 1:skip:end), u(1:skip:end, 1:skip:end), ...
    'k', 'LineWidth', 0.8);

xlabel('Y lattice index');
ylabel('X lattice index');
title('Velocity vector map over velocity magnitude', 'Interpreter', 'none');

saveas(fig, fullfile(pngDir, 'velocity_vector_quiver.png'));
saveas(fig, fullfile(figDir, 'velocity_vector_quiver.fig'));
close(fig);
end

function plot_streamline_cloud(fields, mask, pngDir, figDir)
ensure_dir(pngDir);
ensure_dir(figDir);

u = fields.velocity_x;
v = fields.velocity_y;
speed = fields.velocity_magnitude;
u(mask == 1) = NaN;
v(mask == 1) = NaN;

[X, Y] = meshgrid(1:size(mask, 2), 1:size(mask, 1));

fig = figure('Visible', 'off', 'Color', 'w');
plotData = speed;
plotData(mask == 1) = NaN;
imagesc(plotData);
hold on;
axis image tight;
set(gca, 'YDir', 'normal');
set(gca, 'Color', [0.15, 0.15, 0.15]);
apply_colormap('turbo');
colorbar;

try
    h = streamslice(X, Y, v, u);
    set(h, 'Color', 'k');
catch
    quiver(X, Y, v, u, 'k');
end

xlabel('Y lattice index');
ylabel('X lattice index');
title('Streamline map over velocity magnitude', 'Interpreter', 'none');

saveas(fig, fullfile(pngDir, 'velocity_streamline.png'));
saveas(fig, fullfile(figDir, 'velocity_streamline.fig'));
close(fig);
end

function pltwrite(u_, v_, rho_, U_sqrt, obst2, time, outdir, fileTag)
if nargin < 7 || isempty(outdir)
    outdir = fullfile('output', 'Tecplot_PLT');
end
if nargin < 8 || isempty(fileTag)
    fileTag = ['time_', num2str(time)];
end
ensure_dir(outdir);

filename = fullfile(outdir, ['velocity_', fileTag, '.plt']);
fid = fopen(filename, 'w');

if fid == -1
    fprintf('[Error] Cannot open file: %s\n', filename);
    fprintf('Current folder: %s\n', pwd);
    error('pltwrite: fopen failed.');
end

[lx, ly] = size(obst2);
fprintf(fid, 'VARIABLES="x", "y", "issolid", "u_", "v_", "rho_", "p_", "U_sqrt"\n');
fprintf(fid, 'ZONE T="BOX", I=%d, J=%d, F=POINT\n', ly, lx);
for i = 1:lx
    for j = 1:ly
        fprintf(fid, '%d %d %d %.12e %.12e %.12e %.12e %.12e\n', ...
            i, j, obst2(i, j), u_(i, j), v_(i, j), rho_(i, j), rho_(i, j) / 3.0, U_sqrt(i, j));
    end
end
fclose(fid);
disp(['Tecplot PLT saved: ', filename]);
end

function vtk(u, v, rho, Nx, Ny, U, time, arrgrid, outdir, fileTag)
if nargin < 9 || isempty(outdir)
    outdir = fullfile('output', 'ParaView_VTK');
end
if nargin < 10 || isempty(fileTag)
    fileTag = ['time_', num2str(time)];
end
ensure_dir(outdir);

output_filename = fullfile(outdir, ['velocity_', fileTag, '.vtk']);
output_file = fopen(output_filename, 'w');
if output_file == -1
    error('Cannot open VTK file: %s', output_filename);
end

fprintf(output_file, '# vtk DataFile Version 3.0\n');
fprintf(output_file, 'fluid_state\n');
fprintf(output_file, 'ASCII\n');
fprintf(output_file, 'DATASET RECTILINEAR_GRID\n');
fprintf(output_file, 'DIMENSIONS %d %d 1\n', Nx, Ny);
write_coordinates(output_file, 'X_COORDINATES', Nx);
write_coordinates(output_file, 'Y_COORDINATES', Ny);
fprintf(output_file, 'Z_COORDINATES 1 float\n0\n');
fprintf(output_file, 'POINT_DATA %d\n', Nx * Ny);
write_scalar_field(output_file, 'density', rho);
write_scalar_field(output_file, 'pressure', rho / 3.0);
write_scalar_field(output_file, 'velocity_magnitude', U);
write_scalar_field(output_file, 'solid_mask', arrgrid);
fprintf(output_file, 'VECTORS velocity_vector double\n');
for Y = 1:Ny
    for X = 1:Nx
        fprintf(output_file, '%20.15e %20.15e 0.0\n', u(X, Y), v(X, Y));
    end
end
fclose(output_file);
disp(['VTK saved: ', output_filename]);
end

function write_coordinates(fid, label, dim)
fprintf(fid, '%s %d float\n', label, dim);
fprintf(fid, '%d ', 0:dim - 1);
fprintf(fid, '\n');
end

function write_scalar_field(fid, name, data)
fprintf(fid, 'SCALARS %s double 1\n', name);
fprintf(fid, 'LOOKUP_TABLE default\n');
for Y = 1:size(data, 2)
    for X = 1:size(data, 1)
        fprintf(fid, '%20.15e\n', data(X, Y));
    end
end
end
