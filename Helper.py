import os
import os.path
import numpy as np
import dill
import scipy.io.wavfile
import bob.bio.spear
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib
import bob.measure
import seaborn as sns  
from sklearn import manifold
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from mpl_toolkits.mplot3d import Axes3D


class Helper:

    @staticmethod
    def plot_1dim(scores, labels, file_path):
        """ Plot 1dim distribution"""
        fig1 = plt.figure(1)
        x = np.array(scores)
        x_p = x[np.where(labels == 1)] 
        x_n = x[np.where(labels == 0)] 
        y_p = [0] * len(x_p)
        y_n = [0] * len(x_n)
        
        plt.grid(True)
        plt.scatter(x_p, y_p, marker = '.', color='blue', label='positive', s=4 )
        plt.scatter(x_n, y_n, marker = '.', color='red', label='negative', s=4 )
        fig1.set_size_inches(5.5, 2.1)
        #plt.legend()
        #plt.title(f"Distribution")
        plt.savefig(file_path)
        plt.close()
        return
    
    @staticmethod
    def plot_loss(loss, file_path):
        """ Plot generator and discriminator losses. """

        steps = list(range(len(loss)))
        plt.semilogy(steps, loss)
        plt.legend('Loss')
        plt.title(f"Loss ({len(steps)} steps)")
        plt.savefig(file_path)
        plt.close()
        return
        
    @staticmethod
    def output(filepath, content):
        with open(filepath, 'a') as f:
            f.write(content + '\n')

    @staticmethod
    def load(filepath):
        data = None
        with open(filepath, 'rb') as f:
            data = dill.load(f)
        return data

    @staticmethod
    def save(variable, filepath):
        with open(filepath, 'wb') as f:
            dill.dump(variable, f)
        print(f'data saved successfully at {filepath}')

    @staticmethod
    def create_folder():
        os.makedirs('pic/', exist_ok = True)
        os.makedirs('score/', exist_ok = True)
        os.makedirs('../data/', exist_ok = True)
        return

    @staticmethod
    def get_mfcc(file_path, channel, frame_length, shift, is_vad):
        (sample_rate, signal) =  scipy.io.wavfile.read(file_path)

        if len(signal.shape) == 1:
            single_signal = signal
        else:
            single_signal = signal[:,channel]
        vad = bob.bio.spear.preprocessor.Energy_Thr(
            smoothing_window = 10, \
            win_length_ms = frame_length, \
            win_shift_ms = shift, \
            ratio_threshold = 0.1
        )
        _, _, labels = vad([sample_rate, single_signal.astype('float')])
        if is_vad == False:
            labels = np.ones_like(labels)
        extractor = bob.bio.spear.extractor.Cepstral(
        	n_ceps = 12, \
        	n_filters = 26, \
        	f_min = 0, \
        	f_max = sample_rate / 2, \
        	win_length_ms = frame_length, \
        	win_shift_ms = shift, \
        	pre_emphasis_coef = 0.97, \
        	features_mask = np.arange(0,39), # (12 + 1) * 3
        	normalize_flag = False,
        )
        mfcc = extractor([sample_rate,single_signal.astype('float'), labels]) # 2000,39

        return mfcc

    @staticmethod
    def cal_eer(p_scores, n_scores):
        threshold = bob.measure.eer_threshold(n_scores, p_scores)
        far, frr = bob.measure.farfrr(n_scores, p_scores, threshold)
        eer = max(far, frr) * 100
        return far, frr, eer

    @staticmethod
    def generate_det_curve(p_scores, n_scores, output_path):
        #matplotlib.use('TkAgg')

        plt.switch_backend('agg')    
        bob.measure.plot.det(n_scores, p_scores, 1000, color = (0,0,0), linestyle = '-')
        bob.measure.plot.det_axis([0.01, 99, 0.01, 99])
        threshold = bob.measure.eer_threshold(n_scores, p_scores)
        far, frr = bob.measure.farfrr(n_scores, p_scores, threshold)
        x = range(99)
        ax = plt.gca()
        #ax.plot(x)
        ax.set_aspect('equal', adjustable='box')
        plt.plot([100, -10], [100, -10], linestyle='--', label=f"Equal error rate = {max(far, frr)* 100}%")
        print("##########")
        print(max(far, frr) * 100)
        print("##########")
        plt.xlabel('FAR (%)')
        plt.ylabel('FRR (%)')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend(loc="lower right")
        plt.savefig(output_path)
        plt.cla()
        plt.clf()
        return max(far, frr) * 100

    @staticmethod
    def generate_2det_curve(p_scores, n_scores, p_scores2, n_scores2, output_path):
        #matplotlib.use('TkAgg')

        plt.switch_backend('agg')    
        bob.measure.plot.det(n_scores, p_scores, 1000, color = (1,0,0), linestyle = '-', label="cos-sim")
        bob.measure.plot.det_axis([0.01, 99, 0.01, 99])
        threshold = bob.measure.eer_threshold(n_scores, p_scores)
        far, frr = bob.measure.farfrr(n_scores, p_scores, threshold)
        
        bob.measure.plot.det(n_scores2, p_scores2, 1000, color = (0,1,0), linestyle = '-', label="pairwise")
        bob.measure.plot.det_axis([0.01, 99, 0.01, 99])
        threshold = bob.measure.eer_threshold(n_scores2, p_scores2)
        far, frr = bob.measure.farfrr(n_scores2, p_scores2, threshold)
        x = range(99)
        ax = plt.gca()
        #ax.plot(x)
        #ax.set_aspect('equal', adjustable='box')
        #plt.plot([100, -10], [100, -10], linestyle='--', label=f"Equal error rate = {max(far, frr)* 100}%")
        print("##########")
        print(max(far, frr) * 100)
        print("##########")
        plt.xlabel('FAR (%)')
        plt.ylabel('FRR (%)')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend(loc="lower right")
        plt.savefig(output_path)
        plt.cla()
        plt.clf()
        return max(far, frr) * 100


    @staticmethod
    def generate_multi_det_curve(p_scores, n_scores, output_path, labels):
        #matplotlib.use('TkAgg')
        plt.switch_backend('agg')
        lines = ['-', '--', '-.', ':']
        for p, n, c, l in zip(p_scores, n_scores, lines, labels):
            bob.measure.plot.det(n, p, 1000, linestyle = c, label = l)
            
        bob.measure.plot.det_axis([0.01, 99, 0.01, 99])
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        ax.legend()
        plt.xlabel('FAR (%)')
        plt.ylabel('FRR (%)')
        plt.grid(True)
        plt.savefig(output_path)
        plt.cla()
        plt.clf()


    @staticmethod
    def plot_3d_tsne(embeds, labels, feat_name, output_path):
        print("ploting tsne....")
        # data shape: (n_sample, n_feature)
        plt.switch_backend('agg')
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
    
        r = 1
        pi = np.pi
        cos = np.cos
        sin = np.sin
        phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
        x = r*sin(phi)*cos(theta)
        y = r*sin(phi)*sin(theta)
        z = r*cos(phi)
        ax.plot_surface(
            x, y, z, rstride=1, cstride=1, color='w', alpha=0.3, linewidth=0)

        # tsne = manifold.TSNE(n_components=3, init='pca', random_state=0)
        # X_tsne = tsne.fit_transform(data)
        # ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2],c=labels, cmap=plt.cm.get_cmap("jet", labels[-1]+1))
        ax.scatter(embeds[:,0], embeds[:,1], embeds[:,2], c=labels, s=20)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        # ax.set_aspect("equal")
        plt.title('t-SNE embedding of '+ feat_name)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.clf()

    # def plot(embeds, labels, fig_path='./example.pdf'):

    #     ax = fig.add_subplot(111, projection='3d')

    #     # Create a sphere



    #     plt.savefig(fig_path)

    @staticmethod
    def plot_tsne(data, labels, feat_name, output_path):
        print("ploting tsne....")
        # data shape: (n_sample, n_feature)
        l = len(data["F"])
        plt.switch_backend('agg')
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        tsne = tsne.fit_transform(np.array(data["F"]+data["M"]))
        # plt.title('t-SNE embedding of '+ feat_name)
        plt.scatter(tsne[:l, 0], tsne[:l, 1], c=labels["F"], marker="o", cmap=plt.cm.get_cmap("jet", labels["F"][-1]+1), label="female")
        plt.scatter(tsne[l:, 0], tsne[l:, 1], c=labels["M"], marker="x", cmap=plt.cm.get_cmap("jet", labels["M"][-1]+1), label="male")
        plt.legend()
        plt.savefig(output_path)
        plt.cla()
        plt.clf()

    @staticmethod
    def plot_probabilty_density(sg_p_scores, sg_n_scores, dg_scores, output_path):
        sns.set()

        sns.kdeplot(sg_p_scores, shade=True, color="blue", label="target")
        sns.kdeplot(sg_n_scores, shade=True, color="red", label='same_gender_non-target')
        sns.kdeplot(dg_scores, shade=True, color="grey", label='different_gender_non-target')
        plt.legend()
        plt.xlabel("score")
        plt.ylabel("probability density")
        plt.savefig(output_path)
        plt.cla()
        plt.clf()

    @staticmethod
    def plot_probabilty_density_with_Cprim(p_scores, n_scores, output_path, threshold, fa, fr, Cprim, W):
        sns.set()

        sns.kdeplot(p_scores, shade=True, color="blue", label="target")
        sns.kdeplot(n_scores, shade=True, color="red", label='non-target')
        plt.axvline(threshold, 0, 1, color="green", label=f"threshold: {threshold}\nfalse accept:{fa*100}%\nfalse reject:{fr*100}%")
        plt.axvline(Cthr, 0, 1, color="black", label = f"Cprimary: {Cprim}")
        plt.legend()
        plt.xlabel("score")
        plt.ylabel("probability density")
        plt.savefig(output_path)
        plt.cla()
        plt.clf()

    @staticmethod
    def plot_roc(labels, scores, n_scores, p_scores, output_path):
        threshold = bob.measure.eer_threshold(n_scores, p_scores)
        far, frr = bob.measure.farfrr(n_scores, p_scores, threshold)
        fpr, tpr, roc_thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        lw = 2
        plt.figure(figsize=(10,10))
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
        plt.plot([1, 0], [1, 0], color='navy', lw=lw, linestyle='--', label=f"Equal error rate = {max(far, frr)* 100}%")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # plt.title('ROC curve')
        plt.legend(loc="lower right")
        plt.savefig(output_path)
        plt.cla()
        plt.clf()

    @staticmethod
    def plot_precision_recall(labels, scores, output_path):
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        average_precision = average_precision_score(labels, scores)
        lw = 2
        plt.figure(figsize=(10,10))
        plt.plot(precision, recall, color='darkorange',
                lw=lw, label=f"Average precision = {average_precision*100}%") 
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        # plt.title('Precision recall curve')
        plt.legend(loc="lower right")
        plt.savefig(output_path)
        plt.cla()
        plt.clf()



    @staticmethod
    def check_and_make_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)


    @staticmethod
    def output_eer(scores, labels, context, metrics_path):


        check_and_make_dir(f'{metrics_path}score')
        check_and_make_dir(f'{metrics_path}label')
        check_and_make_dir(f'{metrics_path}pic/det_curve')
        check_and_make_dir(f"{metrics_path}pic/PR")
        check_and_make_dir(f"{metrics_path}pic/PB")
        check_and_make_dir(f"{metrics_path}pic/roc")

        # Output EER to file, and plot DET curve if required.
        save(scores, f'{metrics_path}score/{context}_scores.pkl')
        save(labels, f'{metrics_path}label/{context}_labels.pkl')

        p_scores, n_scores = scores[np.where(labels == True)].astype(np.double), scores[np.where(labels == False)[0]].astype(np.double)
        save(p_scores, f'{metrics_path}score/{context}_pscores.pkl')
        save(n_scores, f'{metrics_path}score/{context}_nscores.pkl')

        # Helper.output('result.txt', '%s, threshold = %f, eer = %f%%, far = %f%%, frr = %f%%\n' % ( \
        # context, threshold, max(far, frr) * 100, far * 100, frr * 100))

        generate_det_curve(p_scores, n_scores, f'{metrics_path}pic/det_curve/{context}_det_curve.png')
        plot_roc(labels=labels, scores=scores, p_scores=p_scores, n_scores=n_scores, output_path=f"{metrics_path}pic/roc/{context}_roc_curve.png")
        plot_probabilty_density(pscores=p_scores, nscores=n_scores, output_path=f"{metrics_path}pic/PB/{context}_prob_den.png")
        plot_precision_recall(pscores=p_scores, nscores=n_scores, output_path=f"{metrics_path}pic/PR/{context}_PR_curve.png")

        return
    

    @staticmethod
    def cal_Cprimary(scores, labels):
        result = []

        total_num = len(scores)

        for i in range(total_num):
            result.append((scores[i], labels[i]))

        sort_result = sorted(result, key=lambda x: x[0])

        count = []
        min_idx = 0
        fr = []
        fa = []
        fr_count = 0
        fa_count = 0
        for i in range(total_num):

            if sort_result[i][1] == 1:
                fr_count += 1
            fr.append(fr_count)

        for i in range(total_num-1, -1, -1):
            if sort_result[i][1] == 0:
                fa_count += 1
            fa.append(fa_count)

        fa.reverse()

        for i in range(total_num):
            current_count = fr[i]+fa[i]
            count.append(current_count)

            if count[min_idx] > current_count:
                min_idx = i


        # print(count[int(min_idx)])
        # print(min_idx)

        threshold = sort_result[min_idx][0]

        fa_count = 0
        fr_count = 0

        for i in range(total_num):
            if sort_result[i][0] <= threshold:
                if sort_result[i][1] == 1:
                    fr_count += 1
            else:
                if sort_result[i][1] == 0:
                    fa_count += 1

        target_prior = 0.05
        Cmiss = 1
        Cfalse = 1
        Cp = []
        Cidx = 0
        for i in range(total_num):
            current_primary = Cmiss * target_prior * fr[i] + Cfalse * (1. - target_prior) * fa[i]
            Cp.append(current_primary)
            if  Cp[Cidx] > current_primary:
                Cidx = i

        Cprimary = Cp[Cidx]/(total_num*target_prior)
        Cthr = sort_result[Cidx][0]

        return threshold, fa_count/total_num, fr_count/total_num, Cprimary, Cthr
