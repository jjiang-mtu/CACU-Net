"""
Rebalancing multi-class loss, Background and 2 Aneurysms
(0) Background
(1) Lumen
(2) Thrombosis
"""

import torch
import torch.nn as nn

num_aneurysm = 2

# Define the weight of each aneurysm, with a weight of 1 for easily segmented aneurysms and a larger weight for poorly segmented aneurysms
aneurysm_weight = [1.0, 2.0]

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_stage1, pred_stage2, target):
        """
        :param pred_stage1: (B, 4, 48, 512, 512)
        :param pred_stage2: (B, 4, 48, 512, 512)
        :param target: (B, 48, 512, 512)
        :return: Dice distance
        """

        # Separate multiple labels for the gold standard
        aneurysm_target = torch.zeros((target.size(0), num_aneurysm, 48, 512, 512))

        for aneurysm_index in range(1, num_aneurysm + 1):
            temp_target = torch.zeros(target.size())
            temp_target[target == aneurysm_index] = 1
            aneurysm_target[:, aneurysm_index - 1, :, :, :] = temp_target

        aneurysm_target = aneurysm_target.cuda()

        # Calculate the loss of the first stage
        dice_stage1_numerator = 0.0  # Numerator of the dice coefficient
        dice_stage1_denominator = 0.0  # Denominator of the dice coefficient

        for aneurysm_index in range(1, num_aneurysm + 1):

            dice_stage1_numerator += 2 * (pred_stage1[:, aneurysm_index, :, :, :] * aneurysm_target[:, aneurysm_index - 1, :, :, :]).sum(dim=1).sum(dim=1).sum(dim=1)

            dice_stage1_numerator *= aneurysm_weight[aneurysm_index - 1]

            dice_stage1_denominator += (pred_stage1[:, aneurysm_index, :, :, :].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                                                aneurysm_target[:, aneurysm_index - 1, :, :, :].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + 1e-5)

            dice_stage1_denominator *= aneurysm_weight[aneurysm_index - 1]

        dice_stage1 = (dice_stage1_numerator / dice_stage1_denominator)

        # Calculate the loss of the second stage
        dice_stage2_numerator = 0.0  # Numerator of the dice coefficient
        dice_stage2_denominator = 0.0  # Denominator of the dice coefficient

        for aneurysm_index in range(1, num_aneurysm + 1):

            dice_stage2_numerator += 2 * (pred_stage2[:, aneurysm_index, :, :, :] * aneurysm_target[:, aneurysm_index - 1, :, :, :]).sum(dim=1).sum(dim=1).sum(dim=1)

            dice_stage2_numerator *= aneurysm_weight[aneurysm_index - 1]

            dice_stage2_denominator += (pred_stage2[:, aneurysm_index, :, :, :].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) +
                                                aneurysm_target[:, aneurysm_index - 1, :, :, :].pow(2).sum(dim=1).sum(dim=1).sum(dim=1) + 1e-5)

            dice_stage2_denominator *= aneurysm_weight[aneurysm_index - 1]

        dice_stage2 = (dice_stage2_numerator / dice_stage2_denominator)

        # Adding together the losses of the two phases
        dice = dice_stage1 + dice_stage2

        # Return dice distance
        return (2 - dice).mean()
