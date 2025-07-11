# -*- coding: utf-8 -*-
def count_paranum(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total para:", total_num, "Trainable para:", trainable_num)
