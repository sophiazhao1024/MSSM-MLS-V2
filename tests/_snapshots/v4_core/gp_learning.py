"""Experimental local-data-efficient surrogate; real physics still decides truth.

Four independent float64 ARD Matérn-3/2 Gaussian processes interpolate physical
observables. The first models mh squared, and the abundance model augments the
three physical input parameters with signed detuning 2*M1 - predicted_mh.
Training uses observed mh for this derived feature; candidate prediction never
uses an uncomputed true mass. M1 is only a bino-mass proxy, not exact m_chi.
Kernel hyperparameters use training marginal likelihood only. The
reported GP spread is model-dependent and uncalibrated, not guaranteed coverage.
An auxiliary fifth GP learns log(HB observed ratio), augmenting its three inputs
with predicted mh. Its uncalibrated threshold probability supplies HB ranking;
when too few valid ratios exist the binary HB classifier is the fallback, never
a second multiplicative penalty. Final acceptance always uses true HB_result.
Neural classifiers separate physical admissibility and numerical success.
A numerical failure is NEVER relabelled physical exclusion.
"""
from dataclasses import asdict
import math
import numpy as np
import torch
from .domain import BOUNDS, CHI2_MAX, HIGGS_INTERVAL, OMEGA_INTERVAL, parameter_array
from .learning import (EnsembleSurrogates, make_network, train_with_holdout,
                       hb_label, _classification_metrics)
from .acquisition import quality_farthest_indices, true_target


def normalized_inputs(rows):
    if not len(rows):
        return np.empty((0, 3), dtype=np.float64)
    x = parameter_array(rows)
    return 2 * (x - BOUNDS[:, 0]) / np.ptp(BOUNDS, axis=1) - 1


def hb_ratio(record):
    """Return only finite positive measured ratios on valid physical rows.

Zero, missing and nonfinite values are unavailable for the logarithmic model;
they are not fabricated exclusion labels. The independent binary fallback uses
known HB_result labels if the ratio regression lacks enough usable observations.
"""
    if record.get('status')!='ok':return None
    diagnostics=record.get('diagnostics')
    value=(diagnostics.get('hb_obsratio') if isinstance(diagnostics,dict) else None)
    if value is None:value=record.get('hb_obsratio')
    if (isinstance(value,bool) or not isinstance(value,(int,float,np.floating))
            or not math.isfinite(value) or value<=0):return None
    return float(value)


def _farthest_indices(x, candidates, count, initial=()):
    """Deterministic farthest-point coverage, with index as tie breaker."""
    candidates = np.asarray(candidates, dtype=int)
    chosen = list(initial)
    active = np.ones(len(candidates), dtype=bool)
    minimum = np.full(len(candidates), np.inf)
    if chosen:
        for i in chosen:
            minimum = np.minimum(minimum, np.sum((x[candidates] - x[i])**2, axis=1))
        active[np.isin(candidates, chosen)] = False
    result = []
    for _ in range(min(count, int(active.sum()))):
        scores = np.where(active, minimum, -np.inf)
        j = int(np.argmax(scores))
        i = int(candidates[j])
        result.append(i)
        active[j] = False
        minimum = np.minimum(minimum, np.sum((x[candidates] - x[i])**2, axis=1))
    return result


def select_anchors(records, limit, target_policy="paper-higgs"):
    """Keep informative near-constraint rows AND global geometric coverage.

All coordinates are taken from this run's physical training data. No archived
targets or externally supplied coordinates are used. A rank-guided half of the
    budget retains the best exact constraint-quality tiers, choosing farthest
    representatives within equal tiers instead of permanently keeping early
    targets. The remainder still covers all physical training data. No quality
    binning, new distance threshold or evaluation-grid information is used.
"""
    if len(records) <= limit:
        return np.arange(len(records), dtype=int)
    x = normalized_inputs([r['x'] for r in records])
    labels = EnsembleSurrogates.transformed_labels(records)
    violation = (np.maximum(HIGGS_INTERVAL[0]-labels[:,1], 0)
                 + np.maximum(labels[:,1]-HIGGS_INTERVAL[1], 0)
                 + np.maximum(labels[:,2]-CHI2_MAX, 0)/10
                 + np.maximum(np.log(OMEGA_INTERVAL[0])-labels[:,3], 0)/.3
                 + np.maximum(labels[:,3]-np.log(OMEGA_INTERVAL[1]), 0)/.3)
    violation += np.array([float(hb_label(r) == 0) for r in records])
    quality = np.column_stack([violation, [not true_target(r, target_policy) for r in records]])
    chosen = quality_farthest_indices(x, quality, limit//2).tolist()
    chosen += _farthest_indices(x, np.arange(len(x)), limit-len(chosen), chosen)
    return np.asarray(chosen, dtype=int)


class ScalarGP:
    """Exact GP on a bounded anchor set, with train-only marginal likelihood."""
    def __init__(self, *, steps=100, input_dim=3):
        if type(steps) is not int or steps < 1:
            raise ValueError('GP steps must be a positive integer')
        self.steps = steps
        if type(input_dim) is not int or input_dim < 1:
            raise ValueError('GP input dimension must be positive')
        self.input_dim = input_dim
        self.x = self.y = self.cholesky = self.alpha = None
        self.theta = None

    def _kernel(self, a, b, theta):
        length = theta[:self.input_dim].exp()
        amplitude = theta[self.input_dim].exp()
        differences = (a[:,None,:] - b[None,:,:]) / length
        r = (differences.square().sum(dim=2) + 1e-16).sqrt()
        root3r = math.sqrt(3)*r
        return amplitude.square() * (1+root3r) * torch.exp(-root3r)

    def _factor(self, theta):
        k = self._kernel(self.x, self.x, theta)
        diagonal = theta[self.input_dim+1].exp().square() + 1e-8
        k = k + torch.eye(len(k), dtype=torch.float64) * diagonal
        # The positive nugget and finite bounded hyperparameters should make
        # this SPD. A failed decomposition is an explicit numerical exception.
        chol = torch.linalg.cholesky(k)
        residual = self.y - theta[self.input_dim+2]
        alpha = torch.cholesky_solve(residual[:,None], chol)
        nll = .5*(residual[:,None]*alpha).sum() + torch.log(torch.diag(chol)).sum()
        nll = nll + .5*len(k)*math.log(2*math.pi)
        return chol, alpha, nll

    def fit(self, x, y):
        self.x = torch.as_tensor(np.asarray(x), dtype=torch.float64).clone()
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.float64).clone()
        if (self.x.ndim != 2 or self.x.shape[1] != self.input_dim or self.y.shape != (len(self.x),)
                or len(self.x) < 2 or not torch.isfinite(self.x).all()
                or not torch.isfinite(self.y).all()):
            raise ValueError('Invalid GP training data')
        theta = torch.tensor([math.log(.5)]*self.input_dim+[0.,math.log(.001),0.],
                             dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([theta], lr=.06)
        best, best_theta = float('inf'), None
        for _ in range(self.steps):
            optimizer.zero_grad(set_to_none=True)
            _, _, loss = self._factor(theta)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite GP marginal likelihood')
            if float(loss.detach()) < best:
                best, best_theta = float(loss.detach()), theta.detach().clone()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                theta[:self.input_dim].clamp_(math.log(.01), math.log(10.))
                theta[self.input_dim].clamp_(math.log(.05), math.log(10.))
                theta[self.input_dim+1].clamp_(math.log(1e-5), math.log(.3))
                theta[self.input_dim+2].clamp_(-5,5)
        with torch.no_grad():
            _, _, final = self._factor(theta)
            if float(final) < best:
                best, best_theta = float(final), theta.detach().clone()
            self.theta = best_theta
            self.cholesky, self.alpha, _ = self._factor(self.theta)
        mean, _ = self.predict(x)
        return {'anchors':len(x),'steps':self.steps,'training_mse_standardized':float(np.mean((mean-np.asarray(y))**2)),
                'negative_log_marginal_likelihood':best,
                'input_dim':self.input_dim,
                'lengthscales':self.theta[:self.input_dim].exp().tolist(),
                'amplitude':float(self.theta[self.input_dim].exp()),
                'noise_standard_deviation':float(self.theta[self.input_dim+1].exp()),
                'mean_standardized':float(self.theta[self.input_dim+2])}

    def predict(self, x):
        a = torch.as_tensor(np.asarray(x), dtype=torch.float64)
        if a.ndim != 2 or a.shape[1] != self.input_dim:
            raise ValueError('GP prediction has wrong input dimension')
        means, deviations = [], []
        with torch.inference_mode():
            for start in range(0, len(a), 2048):
                b = a[start:start+2048]
                k = self._kernel(self.x, b, self.theta)
                means.append((self.theta[self.input_dim+2] + k.T @ self.alpha[:,0]).numpy())
                triangular = torch.linalg.solve_triangular(self.cholesky, k, upper=False)
                variance = self.theta[self.input_dim].exp().square() - triangular.square().sum(dim=0)
                deviations.append(variance.clamp_min(0).sqrt().numpy())
        return ((np.concatenate(means),np.concatenate(deviations)) if means
                else (np.empty(0),np.empty(0)))

    def mean_gradient(self, x):
        """Analytic derivative of posterior mean in standardized output units."""
        a=torch.as_tensor(np.asarray(x),dtype=torch.float64)
        if a.ndim!=2 or a.shape[1]!=self.input_dim:
            raise ValueError('GP derivative has wrong input dimension')
        result=[]
        with torch.inference_mode():
            length=self.theta[:self.input_dim].exp()
            amplitude=self.theta[self.input_dim].exp()
            for start in range(0,len(a),2048):
                b=a[start:start+2048]
                delta=self.x[:,None,:]-b[None,:,:]
                r=((delta/length).square().sum(2)+1e-16).sqrt()
                weight=3*amplitude.square()*torch.exp(-math.sqrt(3)*r)*self.alpha[:,0,None]
                gradient=(weight[:,:,None]*delta/length.square()).sum(0)
                result.append(gradient.numpy())
        return np.concatenate(result) if result else np.empty((0,self.input_dim))

    def export(self):
        return {'steps':self.steps,'input_dim':self.input_dim,'x':self.x,'y':self.y,'theta':self.theta}

    def load_export(self, state):
        if state.get('steps') != self.steps or state.get('input_dim') != self.input_dim:
            raise ValueError('GP step or input-dimension setting mismatch')
        self.x, self.y, self.theta = (state[k].to(dtype=torch.float64) for k in ('x','y','theta'))
        with torch.no_grad():
            self.cholesky,self.alpha,_ = self._factor(self.theta)
        return self


class GPSurrogates:
    """Four observable GPs, optional HB-ratio GP and separate failure models."""
    def __init__(self, settings, seed=0, *, gp_steps=None, max_anchors=None):
        gp_steps=settings.gp_steps if gp_steps is None else gp_steps
        max_anchors=settings.gp_max_anchors if max_anchors is None else max_anchors
        if gp_steps != settings.gp_steps or max_anchors != settings.gp_max_anchors:
            raise ValueError('Put GP overrides in settings so checkpoint identity captures them')
        if (settings.gp_geometry != 'mh2_signed_detuning_hbratio_v3'
                or settings.gp_omega_input_dim != 4 or settings.gp_hb_input_dim != 4):
            raise ValueError('This GP implementation requires the recorded v3 feature geometry')
        if type(seed) is not int or seed < 0 or type(max_anchors) is not int or max_anchors < 4:
            raise ValueError('Invalid GP surrogate configuration')
        self.settings,self.seed,self.fit_count = settings,seed,0
        self.gp_steps,self.max_anchors = gp_steps,max_anchors
        self.regressors=[]
        self.mean=self.scale=None
        self.prediction_scale=None
        self.detuning_mean=self.detuning_scale=None
        self.hb_regressor=None
        self.hb_log_mean=self.hb_log_scale=None
        self.hb_mass_mean=self.hb_mass_scale=None
        self.classifiers={}
        self.constants={}
        torch.set_num_threads(1)

    @property
    def ready(self):
        return len(self.regressors)==4

    def _classifier(self, name, rows, valid, label, seed):
        labels=np.asarray([label(r) for r in rows],dtype=float)
        labels_v=np.asarray([label(r) for r in valid],dtype=float)
        report={'training_count':len(rows),'validation_count':len(valid),
                'positive_count':int(labels.sum()),'negative_count':int(len(labels)-labels.sum())}
        if len(rows)<2 or len(np.unique(labels))<2:
            # Missing class evidence is not sufficient to suppress all unseen
            # points. Treat this optional decision factor as neutral.
            self.constants[name]=1.
            report.update(enabled=False,reason='insufficient_rows_or_single_class',neutral_factor=1.)
            return report
        model=make_network(self.settings,seed)
        stats=train_with_holdout(model, EnsembleSurrogates.inputs([r['x'] for r in rows]),
                                 torch.tensor(labels,dtype=torch.float32),
                                 EnsembleSurrogates.inputs([r['x'] for r in valid]),
                                 torch.tensor(labels_v,dtype=torch.float32),
                                 self.settings,seed,classifier=True)
        self.classifiers[name]=model
        report.update(enabled=True,fit=stats)
        with torch.inference_mode():
            p=torch.sigmoid(model(EnsembleSurrogates.inputs([r['x'] for r in valid]))).squeeze(1).numpy()
        report['validation_metrics']=_classification_metrics(p,labels_v)
        return report

    def fit(self, records):
        train=[r for r in records if not r.get('validation',False)]
        valid=[r for r in records if r.get('validation',False)]
        physical=[r for r in train if r['status']=='ok']
        physical_v=[r for r in valid if r['status']=='ok']
        self.fit_count+=1
        self.regressors,self.classifiers,self.constants=[],{},{}
        self.mean=self.scale=None
        self.prediction_scale=None
        self.detuning_mean=self.detuning_scale=None
        self.hb_regressor=None
        self.hb_log_mean=self.hb_log_scale=None
        self.hb_mass_mean=self.hb_mass_scale=None
        report={'fit':self.fit_count,'ready':False,'model':'ARD_Matern32_GP',
                'physical_train':len(physical),'physical_validation':len(physical_v),
                'max_anchors':self.max_anchors,'gp_steps':self.gp_steps,
                'holdout_scope':'diagnostic fixed holdout, not used in GP likelihood',
                'uncertainty_scope':'model-conditional latent GP standard deviation, not calibrated coverage'}
        if len(physical)<self.settings.min_physical:
            report['reason']='insufficient_physical_training_rows'
            return report
        observed=EnsembleSurrogates.transformed_labels(physical)
        labels=observed.copy()
        labels[:,0]=observed[:,0]**2
        self.mean,self.scale=labels.mean(axis=0),labels.std(axis=0)
        self.scale=np.where(self.scale>1e-12,self.scale,1.)
        self.prediction_scale=observed.std(axis=0)
        self.prediction_scale=np.where(self.prediction_scale>1e-12,self.prediction_scale,1.)
        detuning=2*np.array([r['x'][0] for r in physical])-observed[:,0]
        self.detuning_mean=float(detuning.mean())
        self.detuning_scale=max(float(detuning.std()),1e-12)
        indices=select_anchors(physical,self.max_anchors,self.settings.target_policy)
        x=normalized_inputs([physical[i]['x'] for i in indices])
        y=(labels[indices]-self.mean)/self.scale
        report['anchor_indices_in_physical_training']=indices.tolist()
        report['anchor_eval_ids']=[physical[i].get('eval_id') for i in indices]
        report['anchor_strategy']='exact_quality_true_target_farthest_ties_half_then_global_farthest_half'
        report['gp_geometry']=self.settings.gp_geometry
        report['kernel_input_dimensions']=[3,3,3,4,4]
        report['derived_feature']='signed (2*M1 - mh), true training mh; predicted candidate mh'
        report['detuning_mean'],report['detuning_scale']=self.detuning_mean,self.detuning_scale
        report['mass_transform']='train mh^2; signed-sqrt diagnostic for negative means, GP positive-square factor reduces physical score'
        report['abundance_uncertainty']='conditional GP variance plus first-order mh-error propagation; uncalibrated'
        report['regressors']=[]
        for j,name in enumerate(('mh_squared','mH','chi2_hs','ln_omega')):
            features=(np.column_stack([x,(detuning[indices]-self.detuning_mean)/self.detuning_scale])
                      if j==3 else x)
            model=ScalarGP(steps=self.gp_steps,input_dim=features.shape[1])
            stats=model.fit(features,y[:,j]);stats['label']=name
            self.regressors.append(model);report['regressors'].append(stats)
        ratio_rows=[r for r in physical if hb_ratio(r) is not None]
        ratio_valid=[r for r in physical_v if hb_ratio(r) is not None]
        report['hb_ratio']={'training_count':len(ratio_rows),'validation_count':len(ratio_valid),
                            'unavailable_training_ratio_count':len(physical)-len(ratio_rows),
                            'probability_scope':'uncalibrated GP P(log observed ratio < 0), not final HB truth',
                            'known_label_threshold_disagreements':sum((hb_ratio(r)<=1)!=(hb_label(r)==1)
                                for r in ratio_rows if hb_label(r) is not None)}
        if len(ratio_rows)>=self.settings.min_physical:
            ratio_labels=np.log([hb_ratio(r) for r in ratio_rows])
            self.hb_log_mean,self.hb_log_scale=float(ratio_labels.mean()),float(ratio_labels.std())
            if self.hb_log_scale<=1e-12:self.hb_log_scale=1.
            mass=np.array([r['observables']['mh'] for r in ratio_rows])
            self.hb_mass_mean,self.hb_mass_scale=float(mass.mean()),float(mass.std())
            if self.hb_mass_scale<=1e-12:self.hb_mass_scale=1.
            hi=select_anchors(ratio_rows,self.max_anchors,self.settings.target_policy)
            hx=normalized_inputs([ratio_rows[i]['x'] for i in hi])
            features=np.column_stack([hx,(mass[hi]-self.hb_mass_mean)/self.hb_mass_scale])
            hy=(ratio_labels[hi]-self.hb_log_mean)/self.hb_log_scale
            self.hb_regressor=ScalarGP(steps=self.gp_steps,input_dim=self.settings.gp_hb_input_dim)
            stats=self.hb_regressor.fit(features,hy)
            report['hb_ratio'].update(enabled=True,fit=stats,
                anchor_eval_ids=[ratio_rows[i].get('eval_id') for i in hi],
                mass_feature_mean=self.hb_mass_mean,mass_feature_scale=self.hb_mass_scale,
                logratio_mean=self.hb_log_mean,logratio_scale=self.hb_log_scale)
        else:
            report['hb_ratio'].update(enabled=False,reason='insufficient_finite_positive_ratios',
                                      fallback='binary_HB_classifier_or_neutral_if_unavailable')
        groups={
            'physical':([r for r in train if r['status'] in ('ok','unphysical')],
                        [r for r in valid if r['status'] in ('ok','unphysical')],lambda r:r['status']=='ok'),
            'hb':([r for r in physical if hb_label(r) is not None],
                  [r for r in physical_v if hb_label(r) is not None],lambda r:hb_label(r)),
            # An operational response model, not a physics classification:
            # explicit unphysical outputs are intentionally not its negatives.
            'success':([r for r in train if r['status'] in ('ok','tool_error')],
                       [r for r in valid if r['status'] in ('ok','tool_error')],lambda r:r['status']=='ok')}
        report['classifiers']={}
        for j,(name,(rows,vrows,label)) in enumerate(groups.items()):
            if name=='hb' and self.hb_regressor is not None:
                self.constants['hb']=1.
                report['classifiers'][name]={'enabled':False,'training_count':len(rows),
                    'validation_count':len(vrows),'reason':'HB_ratio_GP_replaces_binary_factor_not_multiplied'}
            else:
                report['classifiers'][name]=self._classifier(name,rows,vrows,label,
                                                            self.seed+self.fit_count*1000+90+j)
        report['ready']=True
        report['output_mean'],report['output_scale']=self.mean.tolist(),self.scale.tolist()
        report['regression_metrics']=[]
        p=self.predict([r['x'] for r in physical])['mean_transformed']
        pv=self.predict([r['x'] for r in physical_v])['mean_transformed']
        labels_v=EnsembleSurrogates.transformed_labels(physical_v)
        for j,name in enumerate(('mh','mH','chi2_hs','ln_omega')):
            good=np.isfinite(p[:,j])
            goodv=np.isfinite(pv[:,j])
            report['regression_metrics'].append({'label':name,
                'training_mse':float(np.mean((p[good,j]-observed[good,j])**2)) if good.any() else None,
                'training_invalid_predictions':int((~good).sum()),
                'validation_mse':float(np.mean((pv[goodv,j]-labels_v[goodv,j])**2)) if goodv.any() else None,
                'validation_invalid_predictions':int((~goodv).sum()),
                'validation_count':len(labels_v)})
        if self.hb_regressor is not None:
            rpred=self.predict([r['x'] for r in ratio_rows])
            rvpred=self.predict([r['x'] for r in ratio_valid])
            report['hb_ratio']['training_logratio_mse']=float(np.mean((rpred['log_hb_ratio_mean']-np.log([hb_ratio(r) for r in ratio_rows]))**2))
            report['hb_ratio']['validation_logratio_mse']=(float(np.mean((rvpred['log_hb_ratio_mean']-np.log([hb_ratio(r) for r in ratio_valid]))**2)) if ratio_valid else None)
            known_valid=[r for r in physical_v if hb_label(r) is not None]
            report['hb_ratio']['binary_validation_metrics']=_classification_metrics(
                self.predict([r['x'] for r in known_valid])['hb_probability'],[hb_label(r) for r in known_valid])
        return report

    def predict(self, rows):
        if not self.ready:
            raise RuntimeError('GP surrogate is not fitted')
        x=normalized_inputs(rows)
        columns=[g.predict(x) for g in self.regressors[:3]]
        first_mean=np.column_stack([a for a,b in columns])*self.scale[:3]+self.mean[:3]
        first_std=np.column_stack([b for a,b in columns])*self.scale[:3]
        mass2=first_mean[:,0]
        positive=mass2>0
        signed_sqrt=lambda v:np.sign(v)*np.sqrt(abs(v))
        mass=signed_sqrt(mass2)
        # Finite symmetric propagation stays finite at the square-root boundary.
        # For a well-positive narrow distribution it approaches sigma_m2/(2*mh).
        # A negative signed mass is only a prediction diagnostic, never a valid
        # physical label; mh_prediction_valid and the score factor expose it.
        sigma_mass=.5*abs(signed_sqrt(mass2+first_std[:,0])-signed_sqrt(mass2-first_std[:,0]))
        z=torch.as_tensor(mass2/np.maximum(first_std[:,0],1e-12),dtype=torch.float64)
        positive_square_probability=(.5*(1+torch.erf(z/math.sqrt(2)))).numpy()
        # This fourth kernel coordinate is a deterministic feature of the three
        # candidate inputs and the first GP prediction. It is not a new oracle
        # input. Signed negative mass predictions are explicitly marked, and
        # their GP positive-square probability penalizes the physical score.
        detuning=2*np.asarray(rows,dtype=float).reshape(-1,3)[:,0]-mass
        omega_inputs=np.column_stack([x,(detuning-self.detuning_mean)/self.detuning_scale])
        omega_z,omega_sigma_z=self.regressors[3].predict(omega_inputs)
        omega_mean=omega_z*self.scale[3]+self.mean[3]
        derivative=-self.regressors[3].mean_gradient(omega_inputs)[:,3]*self.scale[3]/self.detuning_scale
        # Delta method assumes independent first-stage and conditional residuals
        # and local linearity. Thermal resonances and posterior correlation can
        # violate that approximation; no coverage guarantee is claimed.
        sigma_omega=np.sqrt((omega_sigma_z*self.scale[3])**2+(derivative*sigma_mass)**2)
        transformed=np.column_stack([mass,first_mean[:,1],first_mean[:,2],omega_mean])
        std_transformed=np.column_stack([sigma_mass,first_std[:,1],first_std[:,2],sigma_omega])
        std_z=std_transformed/self.prediction_scale
        mean=transformed.copy()
        with np.errstate(over='ignore',under='ignore'):
            mean[:,3]=np.exp(mean[:,3])
        prediction={'mean':mean,'mean_transformed':transformed,'std_standardized':std_z,
                    'std_transformed':std_transformed,
                    'uncertainty':np.sqrt(np.mean(std_z**2,axis=1)),
                    'mh_squared_mean':mass2,'mh_prediction_valid':positive,
                    'mh_square_positive_probability':positive_square_probability,
                    'omega_mh_derivative':derivative}
        tensor=EnsembleSurrogates.inputs(rows)
        with torch.inference_mode():
            for name in ('physical','hb','success'):
                prediction[name+'_probability']=(torch.sigmoid(self.classifiers[name](tensor)).squeeze(1).numpy().astype(float)
                    if name in self.classifiers else np.full(len(rows),self.constants[name]))
        prediction['solver_probability']=prediction['success_probability']
        prediction['physical_probability']*=positive_square_probability
        if self.hb_regressor is not None:
            hb_inputs=np.column_stack([x,(mass-self.hb_mass_mean)/self.hb_mass_scale])
            hb_z,hb_sigma_z=self.hb_regressor.predict(hb_inputs)
            hb_mean=hb_z*self.hb_log_scale+self.hb_log_mean
            hb_derivative=self.hb_regressor.mean_gradient(hb_inputs)[:,3]*self.hb_log_scale/self.hb_mass_scale
            # The same first-order independent-error approximation as omega:
            # conditional GP variance plus sensitivity to predicted-mass error.
            hb_sigma=np.sqrt((hb_sigma_z*self.hb_log_scale)**2+(hb_derivative*sigma_mass)**2)
            z=torch.as_tensor(-hb_mean/np.maximum(hb_sigma,1e-12),dtype=torch.float64)
            prediction['hb_probability']=(.5*(1+torch.erf(z/math.sqrt(2)))).numpy()
            prediction['log_hb_ratio_mean']=hb_mean
            prediction['log_hb_ratio_std']=hb_sigma
            prediction['hb_ratio_mh_derivative']=hb_derivative
        return prediction

    def export(self):
        return {'kind':'mssm_gp_surrogates_v3','settings':asdict(self.settings),
                'seed':self.seed,'fit_count':self.fit_count,'gp_steps':self.gp_steps,'max_anchors':self.max_anchors,
                'mean':None if self.mean is None else self.mean.tolist(),
                'scale':None if self.scale is None else self.scale.tolist(),
                'prediction_scale':None if self.prediction_scale is None else self.prediction_scale.tolist(),
                'detuning_mean':self.detuning_mean,'detuning_scale':self.detuning_scale,
                'gp_geometry':self.settings.gp_geometry,'kernel_input_dimensions':[3,3,3,4,4],
                'regressors':[r.export() for r in self.regressors],
                'hb_regressor':None if self.hb_regressor is None else self.hb_regressor.export(),
                'hb_log_mean':self.hb_log_mean,'hb_log_scale':self.hb_log_scale,
                'hb_mass_mean':self.hb_mass_mean,'hb_mass_scale':self.hb_mass_scale,
                'classifiers':{name:c.state_dict() for name,c in self.classifiers.items()},
                'constants':self.constants}

    def load_export(self,state):
        if (state.get('kind')!='mssm_gp_surrogates_v3' or state.get('settings')!=asdict(self.settings)
                or state.get('seed')!=self.seed or state.get('gp_steps')!=self.gp_steps
                or state.get('max_anchors')!=self.max_anchors):
            raise ValueError('Incompatible GP export')
        self.fit_count=int(state['fit_count'])
        self.mean=None if state['mean'] is None else np.asarray(state['mean'],dtype=float)
        self.scale=None if state['scale'] is None else np.asarray(state['scale'],dtype=float)
        self.prediction_scale=None if state['prediction_scale'] is None else np.asarray(state['prediction_scale'],dtype=float)
        self.detuning_mean,self.detuning_scale=state['detuning_mean'],state['detuning_scale']
        if state.get('gp_geometry')!=self.settings.gp_geometry or state.get('kernel_input_dimensions')!=[3,3,3,4,4]:
            raise ValueError('GP geometry metadata mismatch')
        if len(state['regressors']) not in (0,4):raise ValueError('Expected four observable regressors')
        self.regressors=[ScalarGP(steps=self.gp_steps,input_dim=d).load_export(r)
                         for r,d in zip(state['regressors'],[3,3,3,4])]
        self.hb_regressor=(None if state['hb_regressor'] is None else
                           ScalarGP(steps=self.gp_steps,input_dim=4).load_export(state['hb_regressor']))
        self.hb_log_mean,self.hb_log_scale=state['hb_log_mean'],state['hb_log_scale']
        self.hb_mass_mean,self.hb_mass_scale=state['hb_mass_mean'],state['hb_mass_scale']
        self.classifiers={}
        for name,weights in state['classifiers'].items():
            model=make_network(self.settings,0)
            model.load_state_dict(weights);model.eval();self.classifiers[name]=model
        self.constants=dict(state['constants'])
        return self
