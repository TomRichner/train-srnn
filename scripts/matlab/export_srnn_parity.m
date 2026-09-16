function export_srnn_parity(output_dir)
% Export production MATLAB dynamics in a Python-friendly, nonduplicated layout.
% First run FractionalReservoir/setup_paths in the MATLAB MCP session.
if nargin < 1, output_dir = '/private/tmp/srnn-parity'; end
if ~isfolder(output_dir), mkdir(output_dir); end
preset = 'celltype_pairs_sfaEI_Sc0p2sig0p1_tauSpread0p25_noStim_noise0p025_dualStd_3cond_mu7';
[d, ~, conditions] = srnn_param_preset(preset);
d.sigma_u_noise = 0; d.ode_solver = 'sra1'; d.T_range = [0 40];
[~, revision] = system('git rev-parse HEAD');
for k = 1:numel(conditions)
    cfg = d; condition = conditions{k};
    fields = fieldnames(condition);
    for j = 1:numel(fields)
        if ~strcmp(fields{j}, 'name'), cfg.(fields{j}) = condition.(fields{j}); end
    end
    names = fieldnames(cfg); vals = struct2cell(cfg); args = [names'; vals'];
    model = SRNNCellTypePairs(args{:}); model.build(); p = model.get_params();
    assert(~p.std_zero_floor && ~any(p.n_g_pairs(:)) && all(p.route_scale(:)==1));
    idx = canonical_indices(p);
    fixture = struct('preset',preset,'condition',condition.name,'revision',strtrim(revision), ...
        'W',full(p.W),'a0',p.S_c_vec,'tau_d',p.tau_d,'c',p.c,'S_a',model.S_a, ...
        'n_a',p.n_a,'n_b',p.n_b_pairs(:,1)', ...
        'tau_a_E',p.tau_a_matrix{1},'tau_a_I',p.tau_a_matrix{2}, ...
        'tau_b_rec_E',p.tau_b_rec{1,1},'tau_b_rec_I',p.tau_b_rec{2,1}, ...
        'tau_b_rel_E',p.tau_b_rel{1,1},'tau_b_rel_I',p.tau_b_rel{2,1}, ...
        'state0',model.S0(idx)','h',1/400);
    p.u_interpolant = @(t) zeros(1,p.n);
    rhs = @(t,y) SRNNCellTypePairs.dynamics_fast(t,y,p);
    deriv = rhs(0,model.S0); fixture.derivative0 = deriv(idx)';
    [~, one] = sde_fixed_step(rhs,(0:2)/400,model.S0,[],[],'sra1');
    fixture.state_one_step = one(2,idx);
    fixture.activation_x = linspace(-2,2,p.n)';
    fixture.activation_y = p.activation_function(fixture.activation_x);
    [fixture.rate0,fixture.synaptic0] = outputs(model.S0',p);
    % Non-equilibrium probe exercises c/K and both STD derivatives/products.
    probe=model.S0;
    probe(p.state_layout.x)=linspace(-0.4,0.8,p.n);
    for q=1:2
        probe(p.state_layout.a{q})=linspace(0.05,0.3,numel(p.state_layout.a{q}));
        for post=1:2
            probe(p.state_layout.b{q,post})=linspace(0.3,0.9,numel(p.state_layout.b{q,post}));
        end
    end
    dp=rhs(0,probe); fixture.probe_state=probe(idx)'; fixture.probe_derivative=dp(idx)';
    [fixture.probe_rate,fixture.probe_synaptic]=outputs(probe',p);

    if k == 3
        [t,Y] = sde_fixed_step(rhs,0:1/400:40,model.S0,[],[],'sra1');
        assert_routes(Y,p);
        fixture.time = t(1:4:end); fixture.unforced = Y(1:4:end,idx);
        [fixture.unforced_rate,fixture.unforced_synaptic] = outputs(Y(1:4:end,:),p);
        delta = zeros(size(model.S0)); delta(p.state_layout.x) = 1e-6/sqrt(p.n);
        [~, perturbed] = sde_fixed_step(rhs,0:1/400:40,model.S0+delta,[],[],'sra1');
        fixture.contraction_ratio = norm(perturbed(end,idx)-Y(end,idx))/norm(delta(idx));
        assert(fixture.contraction_ratio < 1,'Example did not contract over 40 seconds');
        fixture.input_vector = 0.1*ones(1,p.n);
        driven = Y(1,:); current = model.S0;
        for block = 1:4
            amplitude = double(block==2 || block==3);
            p.u_interpolant = @(t) amplitude*fixture.input_vector;
            f = @(t,y) SRNNCellTypePairs.dynamics_fast(t,y,p);
            [~, segment] = sde_fixed_step(f,(0:1/400:10),current,[],[],'sra1');
            driven = [driven;segment(5:4:end,:)]; %#ok<AGROW>
            current = segment(end,:)';
        end
        assert_routes(driven,p); fixture.driven = driven(:,idx);
        [fixture.driven_rate,fixture.driven_synaptic] = outputs(driven,p);
        fixture.input_amplitude = double(fixture.time(1:end-1)>=10 & fixture.time(1:end-1)<30);
        % Short unforced convergence test against an independent tight ode45.
        p.u_interpolant = @(t) zeros(1,p.n);
        f = @(t,y) SRNNCellTypePairs.dynamics_fast(t,y,p);
        reference = ode45(f,[0 1],model.S0,odeset('RelTol',1e-11,'AbsTol',1e-13));
        fixture.refinement_hz = [400 800 1600]; fixture.refinement_error = zeros(1,3);
        for q=1:3
            grid=0:1/fixture.refinement_hz(q):1;
            [~, refined]=sde_fixed_step(f,grid,model.S0,[],[],'sra1');
            ref=deval(reference,grid)';
            fixture.refinement_error(q)=max(abs(refined(:,idx)-ref(:,idx)),[],'all');
        end
        assert(all(diff(fixture.refinement_error)<0),'Refinement failed');
    end
    save(fullfile(output_dir,[condition.name '.mat']),'-struct','fixture','-v7');
    fprintf('Exported %s (%d states).\n',condition.name,numel(idx));
end
end

function idx=canonical_indices(p)
idx=[];
for q=1:2
    a=reshape(p.state_layout.a{q},p.n_per_type(q),p.n_a(q));
    idx=[idx reshape(a',1,[])]; %#ok<AGROW>
end
for q=1:2
    assert(isequal(p.tau_b_rec{q,1},p.tau_b_rec{q,2}));
    assert(isequal(p.tau_b_rel{q,1},p.tau_b_rel{q,2}));
    b=reshape(p.state_layout.b{q,1},p.n_per_type(q),p.n_b_pairs(q,1));
    idx=[idx reshape(b',1,[])]; %#ok<AGROW>
end
idx=[idx p.state_layout.x];
end

function assert_routes(Y,p)
for q=1:2
    assert(isequal(Y(:,p.state_layout.b{q,1}),Y(:,p.state_layout.b{q,2})), ...
        'Outgoing route states differ; collapse invalid');
end
end

function [rate,synaptic]=outputs(Y,p)
x=Y(:,p.state_layout.x); gain=ones(size(x));
for q=1:2
    neuron=p.type_indices{q}; n=p.n_per_type(q); na=p.n_a(q); nb=p.n_b_pairs(q,1);
    for j=1:na
        x(:,neuron)=x(:,neuron)-p.c_eff(q)*Y(:,p.state_layout.a{q}((j-1)*n+(1:n)));
    end
    for j=1:nb
        gain(:,neuron)=gain(:,neuron).*Y(:,p.state_layout.b{q,1}((j-1)*n+(1:n)));
    end
end
rate=p.activation_function(x')'; synaptic=rate.*gain;
end
