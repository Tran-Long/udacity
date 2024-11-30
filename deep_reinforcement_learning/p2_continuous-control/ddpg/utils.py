


def hard_update(primary_net, target_net):
    for primary_param, target_param in zip(primary_net.parameters(), target_net.parameters()):
        target_param.data.copy_(primary_param.data)

def soft_update(primary_net, target_net, tau):
    for primary_param, target_param in zip(primary_net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau*primary_param.data + (1-tau)*target_param.data)