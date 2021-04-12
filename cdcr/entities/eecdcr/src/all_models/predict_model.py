import sys
import torch

from cdcr.config import *
from cdcr.entities.eecdcr.src.all_models.model_utils import *

if os.path.isdir("src"):
    # we are running "natively", i.e., with eecdcr folder as working dir
    for pack in os.listdir("src"):
        sys.path.append(os.path.join("src", pack))
    sys.path.append("/src/shared/")
elif os.path.isdir("newsalyze/entities/eecdcr/src"):
    # we are running fomr newsalyze
    for pack in os.listdir("newsalyze/entities/eecdcr/src"):
        sys.path.append(os.path.join("newsalyze/entities/eecdcr/src", pack))
    sys.path.append("/newsalyze/entities/eecdcr/src/shared/")


def init_environment(config):
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    if config.gpu_num != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_num)
        config.use_cuda = True
    else:
        config.use_cuda = False

    config.use_cuda = config.use_cuda and torch.cuda.is_available()

    torch.manual_seed(config.seed)
    if config.use_cuda:
        torch.cuda.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logging.info('Testing with CUDA')

# def read_conll_f1(filename):
#     '''
#     This function reads the results of the CoNLL scorer , extracts the F1 measures of the MUS,
#     B-cubed and the CEAF-e and calculates CoNLL F1 score.
#     :param filename: a file stores the scorer's results.
#     :return: the CoNLL F1
#     '''
#     f1_list = []
#     with open(filename, "r") as ins:
#         for line in ins:
#             new_line = line.strip()
#             if new_line.find('F1:') != -1:
#                 f1_list.append(float(new_line.split(': ')[-1][:-1]))
#
#     muc_f1 = f1_list[1]
#     bcued_f1 = f1_list[3]
#     ceafe_f1 = f1_list[7]
#
#     return (muc_f1 + bcued_f1 + ceafe_f1)/float(3)


# def run_conll_scorer():
#     if config_dict["test_use_gold_mentions"]:
#         event_response_filename = os.path.join(args.out_dir, 'CD_test_event_mention_based.response_conll')
#         entity_response_filename = os.path.join(args.out_dir, 'CD_test_entity_mention_based.response_conll')
#     else:
#         event_response_filename = os.path.join(args.out_dir, 'CD_test_event_span_based.response_conll')
#         entity_response_filename = os.path.join(args.out_dir, 'CD_test_entity_span_based.response_conll')
#
#     event_conll_file = os.path.join(args.out_dir,'event_scorer_cd_out.txt')
#     entity_conll_file = os.path.join(args.out_dir,'entity_scorer_cd_out.txt')
#
#     event_scorer_command = ('perl scorer/scorer.pl all {} {} none > {} \n'.format
#             (config_dict["event_gold_file_path"], event_response_filename, event_conll_file))
#
#     entity_scorer_command = ('perl scorer/scorer.pl all {} {} none > {} \n'.format
#             (config_dict["entity_gold_file_path"], entity_response_filename, entity_conll_file))
#
#     processes = []
#     print('Run scorer command for cross-document event coreference')
#     processes.append(subprocess.Popen(event_scorer_command, shell=True))
#
#     print('Run scorer command for cross-document entity coreference')
#     processes.append(subprocess.Popen(entity_scorer_command, shell=True))
#
#     while processes:
#         status = processes[0].poll()
#         if status is not None:
#             processes.pop(0)
#
#     print ('Running scorers has been done.')
#     print ('Save results...')
#
#     scores_file = open(os.path.join(args.out_dir, 'conll_f1_scores.txt'), 'w')
#
#     event_f1 = read_conll_f1(event_conll_file)
#     entity_f1 = read_conll_f1(entity_conll_file)
#     scores_file.write('Event CoNLL F1: {}\n'.format(event_f1))
#     scores_file.write('Entity CoNLL F1: {}\n'.format(entity_f1))
#
#     scores_file.close()


def test_model(test_set, config, perform_evaluation=True):
    '''
    Loads trained event and entity models and test them on the test set
    :param test_set: a Corpus object, represents the test split
    '''

    init_environment(config)

    device = torch.device("cuda:0" if config.use_cuda else "cpu")

    cd_event_model = load_check_point(config.cd_event_model_path)
    cd_entity_model = load_check_point(config.cd_entity_model_path)

    cd_event_model.to(device)
    cd_entity_model.to(device)

    doc_to_entity_mentions = load_entity_wd_clusters(config)

    EECDCR = EECDCR_PATH.split("/")[-1]
    out_dir = os.path.join(TMP_PATH, EECDCR)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    all_entity_clusters, all_event_clusters, en_f1, ev_f1 = test_models(test_set, cd_event_model, cd_entity_model,
                                                        device, config, write_clusters=True, out_dir=out_dir,
                      doc_to_entity_mentions=doc_to_entity_mentions, analyze_scores=False)

    return all_entity_clusters, all_event_clusters
