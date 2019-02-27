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

def objectiveFunctionRigid(stack, template, netR, params):
    z, tx, ty, dxy = params
    if dxy < 0.6 or dxy > 1.2 or np.any(np.array([z, tx, ty]) > 20) or np.any(np.array([z, tx, ty]) < -20):
        return np.inf
    batch = template.gen_batch(stack, z, tx, ty, dxy, dxy)
    moved, xytheta, cost_cc = netR.run(batch)

    p_consistency = stack.score_param_consistency(xytheta)
    score = -(0.4 * np.mean(cost_cc) + 0.6 * p_consistency)
    if np.isnan(score):
        score = np.inf
    return score

def powell(objFCN, TemplateWarpParams0):
    resultParams = minimize(objFCN, TemplateWarpParams0, method='Powell', options={'xtol': 0.005, 'disp': True})
    return resultParams

def iterativePowellWithRetrain(stack, template, netR, netE, params, niter=5):
    for iter in range(niter):
        res = np.array([3, 3, 3, 0.04]) / (iter + 1)# action noise
        z, tx, ty, dxy = params
        # grid of action noise
        zs, txs, tys, dxys = np.meshgrid(np.linspace(z - res[0], z + res[0], 3), np.linspace(tx - res[1], tx + res[1], 3), np.linspace(ty - res[2], ty + res[2], 3), np.linspace(dxy - res[3], dxy + res[3], 3), sparse=False, indexing='ij')
        n_each = 10
        batch = np.zeros(shape=(810, stack.images[0].shape[0], stack.images[0].shape[1], 2))
        pos = np.array([0, 10])

        # gen batches in best search location + action noise
        for z1, tx1, ty1, dxy1 in zip(zs.flatten(), txs.flatten(), tys.flatten(), dxys.flatten()):
            batch[pos[0]:pos[1], :, :, :] = template.gen_batch(stack, z1, tx1, ty1, dxy1, dxy1, subsample=True, n_subsample=n_each)
            pos += 10

        # retrain nets in new location range
        netR, netE = template.retrain_TF_Both(netR, netE, batch, ntrain=300, nbatch=32)

        # define objective function for search
        def objFCN(params):
            return objectiveFunction(stack, template, netR, netE, params)

        vals = powell(objFCN, params)
        params = vals.x
        print(vals.x, vals.fun)

    return (vals, netR, netE)

def iterativePowellWithRetrainRigid(stack, template, netR, params, niter=5):
    for iter in range(niter):
        res = np.array([3, 3, 3, 0.04]) / (iter + 1)  # action noise
        z, tx, ty, dxy = params
        # grid of action noise
        zs, txs, tys, dxys = np.meshgrid(np.linspace(z - res[0], z + res[0], 3),
                                         np.linspace(tx - res[1], tx + res[1], 3),
                                         np.linspace(ty - res[2], ty + res[2], 3),
                                         np.linspace(dxy - res[3], dxy + res[3], 3), sparse=False, indexing='ij')
        n_each = 10
        batch = np.zeros(shape=(810, stack.images[0].shape[0], stack.images[0].shape[1], 2))
        pos = np.array([0, 10])

        # gen batches in best search location + action noise
        for z1, tx1, ty1, dxy1 in zip(zs.flatten(), txs.flatten(), tys.flatten(), dxys.flatten()):
            batch[pos[0]:pos[1], :, :, :] = template.gen_batch(stack, z1, tx1, ty1, dxy1, dxy1, subsample=True,
                                                               n_subsample=n_each)
            pos += 10

        # retrain nets in new location range
        netR = template.retrain_TF_R(netR, batch, ntrain=300, nbatch=32)

        # define objective function for search
        def objFCN(params):
            return objectiveFunctionRigid(stack, template, netR, params)

        vals = powell(objFCN, params)
        params = vals.x
        print(vals.x, vals.fun)

    return (vals, netR)

