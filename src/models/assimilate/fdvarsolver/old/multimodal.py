"""已废弃,保留用于历史对比,请勿编辑。"""
from functools import partial, lru_cache
import os
import numpy as np
import torch
import torch.nn as nn
import collections.abc
from einops import repeat, rearrange
from src.models.assimilate.utils.varcost import Obs_WeighedL2Norm, Model_Var_Cost, Model_H
import matplotlib
import matplotlib.pyplot as plt

class Solver(nn.Module):
    def __init__(
        self, 
        phi_r, 
        obs_list,
        DA_models, 
        ObsOp_models,
        dt, 
        obs_dir,
    ):
        super(Solver, self).__init__()
        self.phi_r = phi_r
        self.obs_list = obs_list
        H_models = {}
        model_VarCost = {}
        for obs_name in obs_list:
            if obs_name in ["atms", "amsua", "mhs", "hrs4"]:
                obs_err = np.load(os.path.join(obs_dir, f"1b{obs_name}_merged_npy_1.0deg", "avg_obs_error.npz"))
            elif obs_name == "prepbufr":
                obs_err = np.load(os.path.join(obs_dir, "GDAS_prepbufr_merged_npy_1.0deg/obs_sigma.npz"))
            elif obs_name == "satwnd":
                obs_err = np.load(os.path.join(obs_dir, "satwnd_merged_npy_1.0deg/obs_sigma.npz"))
            elif obs_name == "ascat":
                obs_err = np.load(os.path.join(obs_dir, "ascat_b_merged_npy_1.0deg/obs_sigma.npz"))
            obs_err_list = []
            for i, key in enumerate(obs_err.keys()):
                obs_err_list.append(torch.tensor(obs_err[key], dtype=torch.float32, requires_grad=False))
            obs_err = torch.stack(obs_err_list, dim=0)
            H_models[obs_name] = Model_H(obs_err)
            m_NormObs = Obs_WeighedL2Norm(obs_err)
            model_VarCost[obs_name] = Model_Var_Cost(m_NormObs)
        self.DA_models = nn.ModuleDict({
            name: model for name, model in DA_models.items()
        })
        self.ObsOp_models = nn.ModuleDict({
            name: model for name, model in ObsOp_models.items()
        })
        self.H_models = nn.ModuleDict({
            name: model for name, model in H_models.items()
        })
        self.model_VarCost = nn.ModuleDict({
            name: model for name, model in model_VarCost.items()
        })
        self.dt = dt

    def forward(self, xb, sat, sat_mask, std, out_vars):
        return self.solve(xb, sat, sat_mask, std, out_vars)

    def solve(self, xb, sat, sat_mask, std, out_vars):
        xa_dict = self.solver_step(xb, sat, sat_mask, std, out_vars)

        return xa_dict

    def solver_step(self, xb, sat, sat_mask, std, out_vars):
        var_cost_grad_dict = self.var_cost(xb, sat, sat_mask, std, out_vars)
        xas = []
        for obs_name in self.obs_list:
            xa = self.DA_models[obs_name](xb, var_cost_grad_dict[obs_name], out_vars, use_checkpoint=True)
            xas.append(xa)

        return xas

    def var_cost(self, xb, obs, obs_mask, std, out_vars):
        preds = self.forecast(xb, obs[self.obs_list[0]], out_vars)

        var_cost_grad_dict = {}

        for obs_name in self.obs_list:
            B, T, C, H, W = preds.shape
            B, T, Cs, H, W = obs[obs_name].shape

            if obs_name in self.ObsOp_models.keys():
                pred_obs, log_var, tgt_obs = self.ObsOp_models[obs_name](
                    preds.view(B*T, C, H, W), 
                    obs[obs_name].view(B*T, Cs, H, W), 
                    obs_mask[obs_name].view(B*T, 1, H, W), 
                    use_checkpoint=True
                )

                dy, _ = self.H_models[obs_name](
                    pred_obs.view(B, T, -1, H, W), 
                    tgt_obs.view(B, T, -1, H, W), 
                    obs_mask[obs_name].view(B, T, 1, H, W), 
                    std[obs_name],
                )
            elif obs_name == "prepbufr":
                dy, _ = self.H_models[obs_name](
                    preds, 
                    obs[obs_name] * obs_mask[obs_name], 
                    obs_mask[obs_name], 
                    std[obs_name]
                )
            elif obs_name == "satwnd":
                dy, _ = self.H_models[obs_name](
                    preds[:,:,17:43], 
                    obs[obs_name] * obs_mask[obs_name], 
                    obs_mask[obs_name], 
                    std[obs_name]
                )
            elif obs_name == "ascat":
                dy, _ = self.H_models[obs_name](
                    preds[:,:,1:3], 
                    obs[obs_name] * obs_mask[obs_name], 
                    obs_mask[obs_name], 
                    std[obs_name]
                )

            # pred_obserr = std.view(1, 1, -1, 1, 1).to(xb.device, dtype=xb.dtype) * torch.sqrt(torch.exp(log_var.view(B, T, -1, H, W)) * sat_mask)

            # print(f"The observations is {torch.sum(sat_mask) / torch.sum(torch.ones_like(sat_mask)) * 100} %")

            loss = self.model_VarCost[obs_name](dy, std[obs_name])
            # loss = self.model_VarCost(dy, std)
            # print(f"loss.shape is {loss.shape}")

            loss = torch.where(torch.isnan(loss), 0, loss)
            loss = torch.where(torch.isinf(loss), 0, loss)
            
            loss.backward(torch.ones_like(loss), retain_graph=True)
            # torch.nn.utils.clip_grad_value_([xb], clip_value=3.0)
            var_cost_grad = xb.grad.detach()
            # print(f"Norm Grad is {torch.sqrt(torch.mean(var_cost_grad ** 2, dim=(1, 2, 3), keepdim=True))}")
            xb.grad = None
            var_cost_grad = torch.where(torch.isnan(var_cost_grad), 0, var_cost_grad)
            var_cost_grad = torch.where(torch.isinf(var_cost_grad), 0, var_cost_grad)

            normgrad_ = torch.sqrt(torch.mean(var_cost_grad ** 2, dim=(1, 2, 3), keepdim=True))
            # print(f"Norm Grad is {normgrad_}")
            normgrad_ = torch.where(torch.isnan(normgrad_), 1, normgrad_)
            normgrad_ = torch.where(normgrad_ == 0, 1, normgrad_)
            normgrad_ = torch.where(torch.isinf(normgrad_), 1, normgrad_)

            var_cost_grad = var_cost_grad / normgrad_

            # # 将张量展平为一维数组以便绘制直方图
            # var_cost_grad_flat = var_cost_grad[0].double().flatten()

            # # 绘制直方图
            # plt.figure(figsize=(10, 6))
            # sns.histplot(var_cost_grad_flat.cpu().numpy(), bins=50, kde=True, color='blue', alpha=0.3, stat='density')
            # mean_grad = np.mean(var_cost_grad_flat.cpu().numpy())
            # std_grad = np.std(var_cost_grad_flat.cpu().numpy())
            # plt.text(0.02, 0.95, f'Mean: {mean_grad:.3f}\nStd: {std_grad:.3f}',
            #         transform=plt.gca().transAxes, verticalalignment='top',
            #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            # plt.title('Distribution of var_cost_grad')
            # plt.xlabel('Value')
            # plt.ylabel('Frequency')
            # plt.grid(True)

            # # 保存直方图到硬盘
            # plt.savefig('/public02/code/XiChen_1.0deg/figures/var_cost_grad/var_cost_grad_amsua_histogram.png', dpi=300, bbox_inches='tight')  # 保存为 PNG 格式
            # # plt.savefig('var_cost_grad_histogram.pdf')  # 也可以保存为 PDF 格式
            # plt.close()  # 关闭图形以释放内存

            del dy, normgrad_, loss
            torch.cuda.empty_cache()

            var_cost_grad_dict[obs_name] = var_cost_grad

        del preds

        return var_cost_grad_dict

    def forecast(self, x0, yobs, out_vars):
        preds = []
        preds.append(x0)
        for i in range(1, yobs.shape[1]):
            if ((24 // self.dt) > 0) and (i % (24 // self.dt)) == 0:
                # Call the model for 24h forecast
                preds.append(self.phi_r(preds[i - 24 // self.dt],
                                        torch.from_numpy(24 * np.ones((x0.shape[0], 1))).to(x0.device, dtype=torch.float32) / 100,
                                        out_vars,
                                        use_checkpoint=True))
            elif ((12 // self.dt) > 0) and (i % (12 // self.dt)) == 0:
                # Call the model for 24h forecast
                preds.append(self.phi_r(preds[i - 12 // self.dt],
                                        torch.from_numpy(12 * np.ones((x0.shape[0], 1))).to(x0.device, dtype=torch.float32) / 100,
                                        out_vars,
                                        use_checkpoint=True))
            elif ((6 // self.dt) > 0) and (i % (6 // self.dt)) == 0:
                # Call the model for 24h forecast
                preds.append(self.phi_r(preds[i - 6 // self.dt],
                                        torch.from_numpy(6 * np.ones((x0.shape[0], 1))).to(x0.device, dtype=torch.float32) / 100,
                                        out_vars,
                                        use_checkpoint=True))
            elif ((3 // self.dt) > 0) and (i % (3 // self.dt)) == 0:
                # Call the model for 24h forecast
                preds.append(self.phi_r(preds[i - 3 // self.dt],
                                        torch.from_numpy(3 * np.ones((x0.shape[0], 1))).to(x0.device, dtype=torch.float32) / 100,
                                        out_vars,
                                        use_checkpoint=True))
            elif ((1 // self.dt) > 0) and (i % (1 // self.dt)) == 0:
                # Call the model for 24h forecast
                preds.append(self.phi_r(preds[i - 1 // self.dt],
                                        torch.from_numpy(1 * np.ones((x0.shape[0], 1))).to(x0.device, dtype=torch.float32) / 100,
                                        out_vars,
                                        use_checkpoint=True))

        preds = torch.stack(preds, dim=1)

        return preds