""" Holds macros to build attack objects or their evaluatiors simply and
    cleanly. These are meant to be REALLY SIMPLE; if you want to do custom
    things, you'll have to write your own methods to build these
"""


import torch
import torch.optim as optim



# Universal import block
# Block to get the relative imports working
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import math
import prebuilt_loss_functions as plf
import loss_functions as lf
import utils.pytorch_utils as utils
import utils.image_utils as img_utils
import cifar10.cifar_loader as cifar_loader
import cifar10.cifar_resnets as cifar_resnets
import adversarial_training as advtrain
import adversarial_evaluation as adveval
import utils.checkpoints as checkpoints
import adversarial_perturbations as ap
import adversarial_attacks as aa
import spatial_transformers as st


#############################################################################
#                                                                           #
#                   'GLOBAL' KWARGS                                         #
#                                                                           #
#############################################################################

L_INF_BOUND = 8.0 / 255.0
FLOW_LINF = 0.05
TRANS_LINF = 0.05
ROT_LINF = math.pi / 24.0
PGD_ITER = (20, 500)
LOSS_CONVERGENCE = 0.999999
USE_GPU = torch.cuda.is_available()


##############################################################################
#                                                                            #
#                           DELTA L_infinity ATTACKS ONLY                    #
#                                                                            #
##############################################################################

def build_delta_fgsm(model, normalizer, linf_bound=L_INF_BOUND,
                     verbose=False, adv_loss='xentropy', output='attack',
                     manual_gpu=None):

    # Build threat
    delta_threat = ap.ThreatModel(ap.DeltaAddition,
                                  ap.PerturbationParameters(lp_style='inf',
                                                            lp_bound=linf_bound,
                                                            manual_gpu=manual_gpu))

    # Build loss
    assert adv_loss in ['xentropy', 'cw']
    if adv_loss == 'xentropy':
        attack_loss = plf.VanillaXentropy(model, normalizer)
    else:
        cw_loss = lf.CWLossF6(model, normalizer)
        attack_loss = lf.RegularizedLoss({'adv': cw_loss}, {'adv': 1.0})


    # Build attack
    fgsm_attack = aa.FGSM(model, normalizer, delta_threat,
                           attack_loss, manual_gpu=manual_gpu)

    # Return based on output arg
    assert output in ['attack', 'params', 'eval']
    if output == 'attack':
        return fgsm_attack


    attack_kwargs ={'verbose': verbose}
    params = advtrain.AdversarialAttackParameters(fgsm_attack, 1.0,
                        attack_specific_params={'attack_kwargs': attack_kwargs})

    if output == 'params':
        return params


    to_eval= {'top1': 'top1',
              'lpips': 'avg_successful_lpips'}
    eval_result = adveval.EvaluationResult(params,
                                           to_eval=to_eval,
                                           manual_gpu=manual_gpu)
    return eval_result



def build_delta_pgd(model, normalizer, linf_bound=L_INF_BOUND, manual_gpu=None,
                    verbose=False, adv_loss='cw', num_iter=PGD_ITER,
                    loss_convergence=LOSS_CONVERGENCE, output='attack',
                    extra_attack_kwargs=None):

    # Build threat
    delta_threat = ap.ThreatModel(ap.DeltaAddition,
                                  ap.PerturbationParameters(lp_style='inf',
                                                            lp_bound=linf_bound,
                                                            manual_gpu=manual_gpu))

    # Build loss
    assert adv_loss in ['xentropy', 'cw']
    if adv_loss == 'xentropy':
        attack_loss = plf.VanillaXentropy(model, normalizer)
    else:
        cw_loss = lf.CWLossF6(model, normalizer)
        attack_loss = lf.RegularizedLoss({'adv': cw_loss}, {'adv': 1.0},
                                         negate=True)

    # Build attack
    pgd_attack = aa.PGD(model, normalizer, delta_threat,
                         attack_loss, manual_gpu=manual_gpu)

    # Return based on output arg
    assert output in ['attack', 'params', 'eval']
    if output == 'attack':
        return pgd_attack

    extra_attack_kwargs = extra_attack_kwargs or {}
    optimizer = optim.Adam
    optimizer_kwargs = {'lr': 0.01}
    pgd_kwargs = {'num_iterations': num_iter,
                  'signed': False,
                  'optimizer': optimizer,
                  'optimizer_kwargs': optimizer_kwargs,
                  'verbose': verbose,
                  'loss_convergence': loss_convergence}
    pgd_kwargs.update(extra_attack_kwargs)
    params = advtrain.AdversarialAttackParameters(pgd_attack, 1.0,
                           attack_specific_params={'attack_kwargs': pgd_kwargs})

    if output == 'params':
        return params

    to_eval= {'top1': 'top1',
              'lpips': 'avg_successful_lpips'}

    eval_result = adveval.EvaluationResult(params,
                                           to_eval=to_eval, manual_gpu=manual_gpu)
    return eval_result


##############################################################################
#                                                                            #
#                         FLOW/StAdv ATTACKS ONLY                            #
#                                                                            #
##############################################################################

def build_stadv_pgd(model, normalizer, linf_bound=FLOW_LINF, manual_gpu=None,
                    verbose=False, adv_loss='cw', num_iter=PGD_ITER,
                    loss_convergence=LOSS_CONVERGENCE, use_stadv=True,
                    output='attack', norm_hyperparam=0.05,
                    extra_attack_kwargs=None):

    # Build threat
    flow_threat = ap.ThreatModel(ap.ParameterizedXformAdv,
                                 ap.PerturbationParameters(lp_style='inf',
                                                           lp_bound=linf_bound,
                                                   xform_class=st.FullSpatial,
                                                   manual_gpu=manual_gpu,
                                                   use_stadv=use_stadv))


    # Build loss
    assert adv_loss in ['xentropy', 'cw']
    if adv_loss == 'xentropy':
        adv_loss_obj = lf.PartialXentropy(model, normalizer=normalizer)
    else:
        adv_loss_obj = lf.CWLossF6(model, normalizer)

    st_loss = lf.PerturbationNormLoss(lp=2)

    attack_loss = lf.RegularizedLoss({'adv': adv_loss_obj, 'st': st_loss},
                                     {'adv': 1.0, 'st': norm_hyperparam},
                                     negate=True)


    # Build attack
    pgd_attack = aa.PGD(model, normalizer, flow_threat,
                         attack_loss, manual_gpu=manual_gpu)


    # Return based on output arg
    assert output in ['attack', 'params', 'eval']
    if output == 'attack':
        return pgd_attack

    extra_attack_kwargs = extra_attack_kwargs or {}
    optimizer = optim.Adam
    optimizer_kwargs = {'lr': 0.01}
    pgd_kwargs = {'num_iterations': num_iter,
                  'signed': False,
                  'optimizer': optimizer,
                  'optimizer_kwargs': optimizer_kwargs,
                  'verbose': verbose,
                  'loss_convergence': loss_convergence}
    pgd_kwargs.update(extra_attack_kwargs)
    params = advtrain.AdversarialAttackParameters(pgd_attack, 1.0,
                           attack_specific_params={'attack_kwargs': pgd_kwargs})

    if output == 'params':
        return params

    to_eval= {'top1': 'top1',
              'lpips': 'avg_successful_lpips'}

    eval_result = adveval.EvaluationResult(params,
                                           to_eval=to_eval, manual_gpu=manual_gpu)
    return eval_result




##############################################################################
#                                                                            #
#                           Rotation + Translation Only                      #
#                                                                            #
##############################################################################

def build_rot_trans_pgd(model, normalizer, trans_bound=TRANS_LINF,
                        rot_bound=ROT_LINF, manual_gpu=None,
                        verbose=False, adv_loss='cw', num_iter=PGD_ITER,
                        loss_convergence=LOSS_CONVERGENCE,
                        output='attack',
                        extra_attack_kwargs=None):

    # Build threat
    trans_threat = ap.ThreatModel(ap.ParameterizedXformAdv,
                                  ap.PerturbationParameters(lp_style='inf',
                                                        lp_bound=trans_bound,
                                            xform_class=st.TranslationTransform,
                                                            manual_gpu=manual_gpu))
    rotation_threat = ap.ThreatModel(ap.ParameterizedXformAdv,
                                     ap.PerturbationParameters(
                                                xform_class=st.RotationTransform,
                                            lp_style='inf', lp_bound=rot_bound,
                                            manual_gpu=manual_gpu))

    sequence_threat = ap.ThreatModel(ap.SequentialPerturbation,
                                     [trans_threat, rotation_threat])

    # Build loss
    assert adv_loss in ['xentropy', 'cw']
    if adv_loss == 'xentropy':
        attack_loss = plf.VanillaXentropy(model, normalizer)
    else:
        cw_loss = lf.CWLossF6(model, normalizer)
        attack_loss = lf.RegularizedLoss({'adv': cw_loss}, {'adv': 1.0},
                                         negate=True)



    # Build attack
    pgd_attack = aa.PGD(model, normalizer, sequence_threat,
                         attack_loss, manual_gpu=manual_gpu)


    # Return based on output arg
    assert output in ['attack', 'params', 'eval']
    if output == 'attack':
        return pgd_attack

    extra_attack_kwargs = extra_attack_kwargs or {}
    optimizer = optim.Adam
    optimizer_kwargs = {'lr': 0.01}
    pgd_kwargs = {'num_iterations': num_iter,
                  'signed': False,
                  'optimizer': optimizer,
                  'optimizer_kwargs': optimizer_kwargs,
                  'verbose': verbose,
                  'loss_convergence': loss_convergence}
    pgd_kwargs.update(extra_attack_kwargs)
    params = advtrain.AdversarialAttackParameters(pgd_attack, 1.0,
                           attack_specific_params={'attack_kwargs': pgd_kwargs})

    if output == 'params':
        return params

    to_eval= {'top1': 'top1',
              'lpips': 'avg_successful_lpips'}

    eval_result = adveval.EvaluationResult(params,
                                           to_eval=to_eval, manual_gpu=manual_gpu)
    return eval_result



##############################################################################
#                                                                            #
#                     Delta + Rotation + Translation Only                    #
#                                                                            #
##############################################################################


def build_delta_rot_trans_pgd(model, normalizer, delta_bound=L_INF_BOUND,
                              trans_bound=TRANS_LINF,
                              rot_bound=ROT_LINF, manual_gpu=None,
                              verbose=False, adv_loss='cw', num_iter=PGD_ITER,
                              loss_convergence=LOSS_CONVERGENCE,
                              output='attack',
                              extra_attack_kwargs=None):

    # Build threat
    delta_threat = ap.ThreatModel(ap.DeltaAddition,
                                  ap.PerturbationParameters(lp_style='inf',
                                                           lp_bound=delta_bound,
                                                           manual_gpu=manual_gpu))

    trans_threat = ap.ThreatModel(ap.ParameterizedXformAdv,
                                  ap.PerturbationParameters(lp_style='inf',
                                                        lp_bound=trans_bound,
                                            xform_class=st.TranslationTransform,
                                                            manual_gpu=manual_gpu))
    rotation_threat = ap.ThreatModel(ap.ParameterizedXformAdv,
                                     ap.PerturbationParameters(
                                                xform_class=st.RotationTransform,
                                            lp_style='inf', lp_bound=rot_bound,
                                            manual_gpu=manual_gpu))

    sequence_threat = ap.ThreatModel(ap.SequentialPerturbation,
                                     [delta_threat, trans_threat,
                                      rotation_threat])

    # Build loss
    assert adv_loss in ['xentropy', 'cw']
    if adv_loss == 'xentropy':
        attack_loss = plf.VanillaXentropy(model, normalizer)
    else:
        cw_loss = lf.CWLossF6(model, normalizer)
        attack_loss = lf.RegularizedLoss({'adv': cw_loss}, {'adv': 1.0},
                                         negate=True)



    # Build attack
    pgd_attack = aa.PGD(model, normalizer, sequence_threat,
                         attack_loss, manual_gpu=manual_gpu)


    # Return based on output arg
    assert output in ['attack', 'params', 'eval']
    if output == 'attack':
        return pgd_attack

    extra_attack_kwargs = extra_attack_kwargs or {}
    optimizer = optim.Adam
    optimizer_kwargs = {'lr': 0.01}
    pgd_kwargs = {'num_iterations': num_iter,
                  'signed': False,
                  'optimizer': optimizer,
                  'optimizer_kwargs': optimizer_kwargs,
                  'verbose': verbose,
                  'loss_convergence': loss_convergence}
    pgd_kwargs.update(extra_attack_kwargs)
    params = advtrain.AdversarialAttackParameters(pgd_attack, 1.0,
                           attack_specific_params={'attack_kwargs': pgd_kwargs})

    if output == 'params':
        return params

    to_eval= {'top1': 'top1',
              'lpips': 'avg_successful_lpips'}

    eval_result = adveval.EvaluationResult(params,
                                           to_eval=to_eval, use_gpu=USE_GPU)
    return eval_result


##############################################################################
#                                                                            #
#                               Delta + StAdv Attack                         #
#                                                                            #
##############################################################################

def build_delta_stadv_pgd(model, normalizer, delta_bound=L_INF_BOUND,
                          flow_bound=FLOW_LINF, manual_gpu=None,
                          verbose=False, adv_loss='cw', num_iter=PGD_ITER,
                          loss_convergence=LOSS_CONVERGENCE,
                          output='attack',
                          extra_attack_kwargs=None):
    # Build threat
    delta_threat = ap.ThreatModel(ap.DeltaAddition,
                                  ap.PerturbationParameters(lp_style='inf',
                                                            lp_bound=delta_bound,
                                                            manual_gpu=manual_gpu))
    flow_threat = ap.ThreatModel(ap.ParameterizedXformAdv,
                                 ap.PerturbationParameters(lp_style='inf',
                                                           lp_bound=flow_bound,
                                                           xform_class=st.FullSpatial,
                                                           manual_gpu=manual_gpu,
                                                           use_stadv=True))
    sequence_threat = ap.ThreatModel(ap.SequentialPerturbation,
                                 [delta_threat, flow_threat],
                                ap.PerturbationParameters(norm_weights=[0.00, 1.00]))

    # Build loss
    assert adv_loss in ['cw', 'xentropy']
    if adv_loss == 'xentropy':
        adv_loss_obj = lf.PartialXentropy(model, normalizer=normalizer)
        adv_loss_scale = -1.0
    else:
        adv_loss_obj = lf.CWLossF6(model, normalizer)
        adv_loss_scale = 1.0

    st_loss = lf.PerturbationNormLoss(lp=2)

    loss_fxn = lf.RegularizedLoss({'adv': adv_loss_obj, 'st':st_loss},
                                  {'adv': adv_loss_scale, 'st': 0.05},
                                  negate=True)

    # Build attack
    optimizer = optim.Adam
    optimizer_kwargs = {'lr': 0.001}
    pgd_attack = aa.PGD(model, normalizer, sequence_threat, loss_fxn, manual_gpu=manual_gpu)


    assert output in ['attack', 'params', 'eval']
    if output == 'attack':
        return pgd_attack


    pgd_kwargs = {'num_iterations': num_iter,
                  'signed': False,
                  'optimizer': optimizer,
                  'optimizer_kwargs': optimizer_kwargs,
                  'verbose': verbose,
                  'loss_convergence': loss_convergence}

    if extra_attack_kwargs is not None:
        pgd_kwargs.update(extra_attack_kwargs)

    params = advtrain.AdversarialAttackParameters(pgd_attack, 1.0,
                                       attack_specific_params={'attack_kwargs': pgd_kwargs})

    if output == 'params':
        return params

    to_eval= {'top1': 'top1',
              'lpips': 'avg_successful_lpips'}

    eval_result = adveval.EvaluationResult(params,
                                           to_eval=to_eval, manual_gpu=manual_gpu)
    return eval_result




##############################################################################
#                                                                            #
#                           Delta + StAdv + Rot + Trans                      #
#                                                                            #
##############################################################################

def build_delta_stadv_rot_trans_pgd(model, normalizer, delta_bound=L_INF_BOUND,
                                   flow_bound=FLOW_LINF, trans_bound=TRANS_LINF,
                                   rot_bound=ROT_LINF, manual_gpu=None,
                              verbose=False, adv_loss='cw', num_iter=PGD_ITER,
                              loss_convergence=LOSS_CONVERGENCE,
                              output='attack',
                              extra_attack_kwargs=None):
    # Build threat
    delta_threat = ap.ThreatModel(ap.DeltaAddition,
                                  ap.PerturbationParameters(lp_style='inf',
                                                            lp_bound=delta_bound,
                                                            manual_gpu=manual_gpu))
    flow_threat = ap.ThreatModel(ap.ParameterizedXformAdv,
                                 ap.PerturbationParameters(lp_style='inf',
                                                           lp_bound=flow_bound,
                                                           xform_class=st.FullSpatial,
                                                           manual_gpu=manual_gpu,
                                                           use_stadv=True))

    trans_threat = ap.ThreatModel(ap.ParameterizedXformAdv,
                                  ap.PerturbationParameters(lp_style='inf',
                                                        lp_bound=trans_bound,
                                            xform_class=st.TranslationTransform,
                                                            manual_gpu=manual_gpu))
    rotation_threat = ap.ThreatModel(ap.ParameterizedXformAdv,
                                     ap.PerturbationParameters(
                                                xform_class=st.RotationTransform,
                                            lp_style='inf', lp_bound=rot_bound,
                                            manual_gpu=manual_gpu))


    sequence_threat = ap.ThreatModel(ap.SequentialPerturbation,
                                 [delta_threat, flow_threat, trans_threat,
                                  rotation_threat],
                                ap.PerturbationParameters(norm_weights=[0.00,
                                                            1.00, 1.0, 1.0]))

    # Build loss
    assert adv_loss in ['cw', 'xentropy']
    if adv_loss == 'xentropy':
        adv_loss_obj = lf.PartialXentropy(model, normalizer=normalizer)
        loss_multi = -1.0
    else:
        adv_loss_obj = lf.CWLossF6(model, normalizer)
        loss_multi = 1.0

    st_loss = lf.PerturbationNormLoss(lp=2)

    loss_fxn = lf.RegularizedLoss({'adv': adv_loss_obj, 'st':st_loss},
                                  {'adv': loss_multi, 'st': 0.05},
                                  negate=True)

    # Build attack
    optimizer = optim.Adam
    optimizer_kwargs = {'lr': 0.001}
    pgd_attack = aa.PGD(model, normalizer, sequence_threat, loss_fxn,
                         manual_gpu=manual_gpu)

    assert output in ['attack', 'params', 'eval']
    if output == 'attack':
        return pgd_attack

    pgd_kwargs = {'num_iterations': num_iter,
                  'signed': False,
                  'optimizer': optimizer,
                  'optimizer_kwargs': optimizer_kwargs,
                  'verbose': verbose,
                  'loss_convergence': loss_convergence}
    if extra_attack_kwargs is not None:
        pgd_kwargs.update(extra_attack_kwargs)

    params = advtrain.AdversarialAttackParameters(pgd_attack, 1.0,
                                       attack_specific_params={'attack_kwargs': pgd_kwargs})
    if output == 'params':
        return params

    to_eval= {'top1': 'top1',
              'lpips': 'avg_successful_lpips'}

    eval_result = adveval.EvaluationResult(params,
                                           to_eval=to_eval, manual_gpu=manual_gpu)
    return eval_result






def build_stadv_rot_trans_pgd(model, normalizer,
                                   flow_bound=FLOW_LINF, trans_bound=TRANS_LINF,
                                   rot_bound=ROT_LINF, manual_gpu=None,
                              verbose=False, adv_loss='cw', num_iter=PGD_ITER,
                              loss_convergence=LOSS_CONVERGENCE,
                              output='attack',
                              extra_attack_kwargs=None):
    # Build threat

    flow_threat = ap.ThreatModel(ap.ParameterizedXformAdv,
                                 ap.PerturbationParameters(lp_style='inf',
                                                           lp_bound=flow_bound,
                                                           xform_class=st.FullSpatial,
                                                           manual_gpu=manual_gpu,
                                                           use_stadv=True))

    trans_threat = ap.ThreatModel(ap.ParameterizedXformAdv,
                                  ap.PerturbationParameters(lp_style='inf',
                                                        lp_bound=trans_bound,
                                            xform_class=st.TranslationTransform,
                                                            manual_gpu=manual_gpu))
    rotation_threat = ap.ThreatModel(ap.ParameterizedXformAdv,
                                     ap.PerturbationParameters(
                                                xform_class=st.RotationTransform,
                                            lp_style='inf', lp_bound=rot_bound,
                                            manual_gpu=manual_gpu))


    sequence_threat = ap.ThreatModel(ap.SequentialPerturbation,
                                 [flow_threat, trans_threat,
                                  rotation_threat],
                                ap.PerturbationParameters(norm_weights=[1.00,
                                                            1.00, 1.0, 1.0]))

    # Build loss
    assert adv_loss in ['cw', 'xentropy']
    if adv_loss == 'xentropy':
        adv_loss_obj = lf.PartialXentropy(model, normalizer=normalizer)
    else:
        adv_loss_obj = lf.CWLossF6(model, normalizer)

    st_loss = lf.PerturbationNormLoss(lp=2)

    loss_fxn = lf.RegularizedLoss({'adv': adv_loss_obj, 'st':st_loss},
                                  {'adv': 1.0, 'st': 0.05},
                                  negate=True)

    # Build attack
    optimizer = optim.Adam
    optimizer_kwargs = {'lr': 0.001}
    pgd_attack = aa.PGD(model, normalizer, sequence_threat, loss_fxn,
                         manual_gpu=manual_gpu)

    assert output in ['attack', 'params', 'eval']
    if output == 'attack':
        return pgd_attack

    pgd_kwargs = {'num_iterations': num_iter,
                  'signed': False,
                  'optimizer': optimizer,
                  'optimizer_kwargs': optimizer_kwargs,
                  'verbose': verbose,
                  'loss_convergence': loss_convergence}
    if extra_attack_kwargs is not None:
        pgd_kwargs.update(extra_attack_kwargs)

    params = advtrain.AdversarialAttackParameters(pgd_attack, 1.0,
                                       attack_specific_params={'attack_kwargs': pgd_kwargs})
    if output == 'params':
        return params

    to_eval= {'top1': 'top1',
              'lpips': 'avg_successful_lpips'}

    eval_result = adveval.EvaluationResult(params,
                                           to_eval=to_eval, manual_gpu=manual_gpu)
    return eval_result

