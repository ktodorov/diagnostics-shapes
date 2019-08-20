import pickle

import torch
from tensorboardX import SummaryWriter

from metrics.rsa import representation_similarity_analysis
from metrics.entropy import language_entropy


class MetricsHelper:
    def __init__(self, run_folder, seed, vocab_size):
        self._writer = SummaryWriter(logdir=run_folder + "/" + str(seed))
        self._run_folder = run_folder
        self._best_valid_acc = -1
        self._running_loss = 0.0
        self._vocab_size = vocab_size

    def log_metrics(
        self,
        model,
        valid_meta_data,
        valid_features,
        valid_loss_meter,
        valid_acc_meter,
        valid_entropy_meter,
        valid_messages,
        hidden_sender,
        hidden_receiver,
        loss,
        i):

        # self._running_loss += loss
        if type(valid_loss_meter) == list:
            average_loss = sum([list_item.avg for list_item in valid_loss_meter]) / len(valid_loss_meter)
        else:
            average_loss = valid_loss_meter.avg

        if type(valid_acc_meter) == list:
            average_accuracy = sum([list_item.avg for list_item in valid_acc_meter]) / len(valid_acc_meter)
        else:
            average_accuracy = valid_acc_meter.avg

        num_unique_messages = len(torch.unique(valid_messages, dim=0))
        valid_messages = valid_messages.cpu().numpy()

        rsa_sr, rsa_si, rsa_ri, rsa_sm, topological_similarity, pseudo_tre = representation_similarity_analysis(
            valid_features,
            valid_meta_data,
            valid_messages,
            hidden_sender,
            hidden_receiver,
            vocab_size=self._vocab_size,
            tre=True,
        )
        l_entropy = language_entropy(valid_messages)

        if self._writer is not None:
            self._writer.add_scalar("avg_loss", average_loss, i)
            self._writer.add_scalar("avg_convergence",
                                    self._running_loss / (i + 1), i)
            self._writer.add_scalar("avg_acc", average_accuracy, i)
            # self._writer.add_scalar(
            #     "avg_entropy", valid_entropy_meter.avg, i)
            self._writer.add_scalar("avg_unique_messages",
                                    num_unique_messages, i)
            self._writer.add_scalar("rsa_sr", rsa_sr, i)
            self._writer.add_scalar("rsa_si", rsa_si, i)
            self._writer.add_scalar("rsa_ri", rsa_ri, i)
            self._writer.add_scalar("rsa_sm", rsa_sm, i)
            self._writer.add_scalar(
                "topological_similarity", topological_similarity, i
            )
            self._writer.add_scalar("pseudo_tre", pseudo_tre, i)
            self._writer.add_scalar("language_entropy", l_entropy, i)

        if average_accuracy > self._best_valid_acc:
            self._best_valid_acc = average_accuracy
            torch.save(model.state_dict(),
                       "{}/best_model".format(self._run_folder))

        metrics = {
            "loss": average_loss,
            "acc": average_accuracy,
            # "entropy": valid_entropy_meter.avg,
            "l_entropy": l_entropy,
            "rsa_sr": rsa_sr,
            "rsa_si": rsa_si,
            "rsa_ri": rsa_ri,
            "rsa_sm": rsa_sm,
            "pseudo_tre": pseudo_tre,
            "topological_similarity": topological_similarity,
            "num_unique_messages": num_unique_messages,
            "avg_convergence": self._running_loss / (i + 1),
        }
        # dump metrics
        pickle.dump(
            metrics, open(
                "{}/metrics_at_{}.p".format(self._run_folder, i), "wb")
        )
