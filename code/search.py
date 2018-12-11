from scipy.optimize import minimize, basinhopping
import numpy as np

def objectiveFunction(stack, template, netR, netE, params):
    z, tx, ty, dxy = params
    if dxy < 0.6 or dxy > 1.2 or np.any(np.array([z, tx, ty]) > 20) or np.any(np.array([z, tx, ty]) < -20):
        return np.inf
    batch = template.gen_batch(stack, z, tx, ty, dxy, dxy)
    moved, xytheta, _ = netR.run(batch)
    for i in range(moved.shape[0]):
        batch[i, :, :, 1] = np.squeeze(moved[i, :, :])

    tformed, theta, cost_cc, cost, cost2 = netE.run(batch)
    p_consistency = stack.score_param_consistency(xytheta)
    score = -(0.4 * np.mean(cost_cc) + 0.6 * p_consistency - cost2)
    if np.isnan(score):
        score = np.inf
    return score


class ScaledStep(object):

    def __call__(self, x):
        print('ScaledStep')
        x[0] += np.random.uniform(-2.0, 2.0)
        x[1] += np.random.uniform(-2.0, 2.0)
        x[2] += np.random.uniform(-2.0, 2.0)
        x[3] += np.random.uniform(-0.05, 0.05)
        return x


def nelderMeadSimplex(objFCN, TemplateWarpParams0):
    resultParams = minimize(objFCN, TemplateWarpParams0, method='nelder-mead', options={'xtol': 0.025, 'disp': True})
    return resultParams


def powell(objFCN, TemplateWarpParams0):
    resultParams = minimize(objFCN, TemplateWarpParams0, method='Powell', options={'xtol': 0.005, 'disp': True})
    return resultParams


def bfgs(objFCN, TemplateWarpParams0):
    resultParams = minimize(objFCN, TemplateWarpParams0, method='BFGS', options={'eps': [0.5, 0.5, 0.5, 0.025], 'disp': True})
    return resultParams


def bhopping(objFCN, TemplateWarpParams0):
    step = ScaledStep()
    minimizer_kwargs = {'method': 'Powell', 'options': {'xtol': 0.01, 'disp': True}}
    resultParams = basinhopping(objFCN, TemplateWarpParams0, minimizer_kwargs=minimizer_kwargs, take_step=step)
    return resultParams


def iterativePowellWithRetrain(stack, template, netR, netE, params, niter=5):
    for iter in range(niter):
        res = np.array([3, 3, 3, 0.04]) / (iter + 1)
        z, tx, ty, dxy = params
        zs, txs, tys, dxys = np.meshgrid(np.linspace(z - res[0], z + res[0], 3), np.linspace(tx - res[1], tx + res[1], 3), np.linspace(ty - res[2], ty + res[2], 3), np.linspace(dxy - res[3], dxy + res[3], 3), sparse=False, indexing='ij')
        n_each = 10
        batch = np.zeros(shape=(810, 265, 257, 2))
        pos = np.array([0, 10])
        for z1, tx1, ty1, dxy1 in zip(zs.flatten(), txs.flatten(), tys.flatten(), dxys.flatten()):
            batch[pos[0]:pos[1], :, :, :] = template.gen_batch(stack, z1, tx1, ty1, dxy1, dxy1, subsample=True, n_subsample=n_each)
            pos += 10

        netR, netE = template.retrain_TF_Both(netR, netE, batch, ntrain=300, nbatch=32)

        def objFCN(params):
            return objectiveFunction(stack, template, netR, netE, params)

        vals = powell(objFCN, params)
        params = vals.x
        print(vals.x, vals.fun)

    return (vals, netR, netE)


def param_search(self, IMG, zlim, theta_xlim, theta_ylim, xy_res):
    nspacing = 8
    nbatch = nspacing ** 4
    n_each_param = 1
    ntrain = [0]
    print('loading models...')
    fsys.cd('D:/__Atlas__')
    params = {'load_file': 'D:/__Atlas__/model_saves/model-Rigid30890', 
     'save_file': 'regNETshallow', 
     'save_interval': 1000, 
     'batch_size': 32, 
     'lr': 0.0001, 
     'rms_decay': 0.9, 
     'rms_eps': 1e-08, 
     'width': 265, 
     'height': 257, 
     'numParam': 3, 
     'train': True}
    netR = RigidNet(params)
    params['load_file'] = 'D:/__Atlas__/model_saves/model-Elastic30890'
    params['save_file'] = 'Elastic2'
    netE = TpsNet(params)
    param_dist = []
    max_score = 0
    max_param = [0, 0, 0]
    for res in range(len(ntrain)):
        print('\ngenerating training batch...')
        z_vals = []
        theta_x_vals = []
        theta_y_vals = []
        dxy_vals = []
        zs = np.linspace(zlim[0], zlim[1], nspacing)
        txs = np.linspace(theta_xlim[0], theta_xlim[1], nspacing)
        tys = np.linspace(theta_ylim[0], theta_ylim[1], nspacing)
        dxys = np.linspace(xy_res[0], xy_res[1], nspacing)
        for z in zs:
            for tx in txs:
                for ty in tys:
                    for dxy in dxys:
                        z_vals.append(z)
                        theta_x_vals.append(tx)
                        theta_y_vals.append(ty)
                        dxy_vals.append(dxy)

        if ntrain[res] > 0:
            big_batch = np.zeros(shape=(n_each_param * nbatch, params['width'], params['height'], 2), dtype=np.float32)
            pos = (
             0, n_each_param)
            for z, tx, ty, dxy in tqdm.tqdm(zip(z_vals, theta_x_vals, theta_y_vals, dxy_vals)):
                big_batch[pos[0]:pos[1], :, :, :] = self.gen_batch(IMG, z, tx, ty, dxy, dxy, subsample=True, n_subsample=n_each_param)
                pos = (pos[0] + n_each_param, pos[1] + n_each_param)

            netR, netE = self.retrain_TF_Both(netR, netE, big_batch, ntrain=ntrain[res], nbatch=32)
        print('\ncomputing parameter scores...')
        score = np.zeros(shape=(nbatch,))
        each_score = np.zeros(shape=(nbatch, 3))
        pos = 0
        for z, tx, ty, dxy in tqdm.tqdm(zip(z_vals, theta_x_vals, theta_y_vals, dxy_vals)):
            batch = self.gen_batch(IMG, z, tx, ty, dxy, dxy)
            tformed, xytheta, _ = netR.run(batch)
            for i in range(tformed.shape[0]):
                batch[i, :, :, 1] = np.squeeze(tformed[i, :, :])

            tformed, theta, cost_cc, cost, cost2 = netE.run(batch)
            p_consistency = IMG.score_param_consistency(xytheta)
            score[pos] = 0.4 * np.mean(cost_cc) + 0.6 * p_consistency - cost2
            if np.isnan(score[pos]):
                score[pos] = 0
            each_score[(pos, 0)] = cost_cc
            each_score[(pos, 1)] = p_consistency
            each_score[(pos, 2)] = cost2
            pos += 1

        param_dist.append(zip(z_vals, theta_x_vals, theta_y_vals, dxy_vals, score, each_score.T.tolist()[0], each_score.T.tolist()[1], each_score.T.tolist()[2]))
        plt.figure()
        n, bins, _ = plt.hist(score)
        plt.show()
        max_id = np.argmax(score)
        print('\nmax score:', np.max(score), 'pos:', z_vals[max_id], theta_x_vals[max_id], theta_y_vals[max_id], dxy_vals[max_id])
        if np.max(score) > max_score:
            max_score = np.max(score)
            max_param = [z_vals[max_id], theta_x_vals[max_id], theta_y_vals[max_id], dxy_vals[max_id]]
            z_span = np.asscalar(np.diff(zlim)) / 4.0
            zlim = [z_vals[max_id] - z_span, z_vals[max_id] + z_span]
            tx_span = np.asscalar(np.diff(theta_xlim)) / 4.0
            theta_xlim = [theta_x_vals[max_id] - tx_span, theta_x_vals[max_id] + tx_span]
            ty_span = np.asscalar(np.diff(theta_ylim)) / 4.0
            theta_ylim = [theta_y_vals[max_id] - ty_span, theta_y_vals[max_id] + ty_span]
            dxy_span = np.asscalar(np.diff(xy_res)) / 4.0
            xy_res = [dxy_vals[max_id] - dxy_span, dxy_vals[max_id] + dxy_span]

    tf.reset_default_graph()
    del netR
    del netE
    return (max_score, max_param, param_dist)