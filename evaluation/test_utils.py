import os
import numpy as np
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt

def make_output_folders(ROOT, prefix):
    os.makedirs(
        "{}/results_{}/scores/".format(ROOT, prefix),
        exist_ok=True,
    )
    os.makedirs(
        "{}/results_{}/figures/".format(ROOT, prefix),
        exist_ok=True,
    )
    os.makedirs(
        "{}/tensors/".format(ROOT),
        exist_ok=True,
    )

def make_score_file(nn, out_dataset, filename):
    make_output_folders(nn, out_dataset)
    return


def write_score_file(ROOT, prefix, method, dataset, data):
    os.makedirs(
        "{}/results_{}/scores/{}/".format(ROOT, prefix, method),
        exist_ok=True,
    )
    f = open(
        "{}/results_{}/scores/{}/{}.txt".format(ROOT, prefix, method, dataset),
        "w",)
    np.savetxt(f, data, delimiter=",")
    f.close()

def vis_score(ROOT, prefix, method, in_name, out_name, score_in,score_ood, make_plot = True,add_to_title=None,swap_classes=False):
  os.makedirs(
        "{}/results_{}/figures/{}/".format(ROOT, prefix, method),
        exist_ok=True,
  )
  vis_path = "{}/results_{}/figures/{}/{}_vs_{}.png".format(ROOT, prefix, method, in_name, out_name)
  score_in = score_in.reshape((-1, 1))
  score_ood = score_ood.reshape((-1, 1))

  num_in = score_in.shape[0]
  num_out = score_ood.shape[0]

  onehots = np.zeros(num_in + num_out, dtype=np.int32)
  onehots[:num_in] += 1

  scores = np.squeeze(np.vstack((score_in, score_ood)))

  auroc = roc_auc_score(onehots, scores)

  to_replot_dict = dict()

  if swap_classes == False:
    out_scores,in_scores = scores[onehots==0], scores[onehots==1]
  else:
    out_scores,in_scores = scores[onehots==1], scores[onehots==0]

  if make_plot:
    plt.figure(figsize = (5.5,3),dpi=100)

    if add_to_title is not None:
      plt.title(add_to_title+" AUROC="+str(float(auroc*100))[:6]+"%",fontsize=14)
    else:
      plt.title(" AUROC="+str(float(auroc*100))[:6]+"%",fontsize=14)


  vals,bins = np.histogram(in_scores,bins = 51)
  bin_centers = (bins[1:]+bins[:-1])/2.0

  if make_plot:
    plt.plot(bin_centers,vals,linewidth=4,color="navy",marker="",label="in:" + in_name)
    plt.fill_between(bin_centers,vals,[0]*len(vals),color="navy",alpha=0.3)

  to_replot_dict["out_bin_centers"] = bin_centers
  to_replot_dict["out_vals"] = vals

  vals,bins = np.histogram(out_scores,bins = 51)
  bin_centers = (bins[1:]+bins[:-1])/2.0

  if make_plot:
    plt.plot(bin_centers,vals,linewidth=4,color="crimson",marker="",label="out:" + out_name)
    plt.fill_between(bin_centers,vals,[0]*len(vals),color="crimson",alpha=0.3)

  to_replot_dict["in_bin_centers"] = bin_centers
  to_replot_dict["in_vals"] = vals

  if make_plot:
    plt.xlabel("Score",fontsize=14)
    plt.ylabel("Count",fontsize=14)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.ylim([0,None])

    plt.legend(fontsize = 14)

    plt.tight_layout()
    # plt.show()
    plt.savefig(vis_path)
  return auroc,to_replot_dict
