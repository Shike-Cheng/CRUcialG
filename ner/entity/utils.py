import numpy as np
import json
import logging
import re
logger = logging.getLogger('root')

def ioc_reg(span):
    IoC_regex = {
        "SO": [
            r"([a-z]{3,}\:\/\/[\S]{16,})",
            r"(([a-z0-9\-]{2,}\[?\.\]?)+(abogado|ac|academy|accountants|active|actor|ad|adult|ae|aero|af|ag|agency|ai|airforce|al|allfinanz|alsace|am|amsterdam|an|android|ao|aq|aquarelle|ar|archi|army|arpa|as|asia|associates|at|attorney|au|auction|audio|autos|aw|ax|axa|az|ba|band|bank|bar|barclaycard|barclays|bargains|bayern|bb|bd|be|beer|berlin|best|bf|bg|bh|bi|bid|bike|bingo|bio|biz|bj|black|blackfriday|bloomberg|blue|bm|bmw|bn|bnpparibas|bo|boo|boutique|br|brussels|bs|bt|budapest|build|builders|business|buzz|bv|bw|by|bz|bzh|ca|cal|camera|camp|cancerresearch|canon|capetown|capital|caravan|cards|care|career|careers|cartier|casa|cash|cat|catering|cc|cd|center|ceo|cern|cf|cg|ch|channel|chat|cheap|christmas|chrome|church|ci|citic|city|ck|cl|claims|cleaning|click|clinic|clothing|club|cm|cn|co|coach|codes|coffee|college|cologne|com|community|company|computer|condos|construction|consulting|contractors|cooking|cool|coop|country|cr|credit|creditcard|cricket|crs|cruises|cu|cuisinella|cv|cw|cx|cy|cymru|cz|dabur|dad|dance|dating|day|dclk|de|deals|degree|delivery|democrat|dental|dentist|desi|design|dev|diamonds|diet|digital|direct|directory|discount|dj|dk|dm|dnp|do|docs|domains|doosan|durban|dvag|dz|eat|ec|edu|education|ee|eg|email|emerck|energy|engineer|engineering|enterprises|equipment|er|es|esq|estate|et|eu|eurovision|eus|events|everbank|exchange|expert|exposed|fail|farm|fashion|feedback|fi|finance|financial|firmdale|fish|fishing|fit|fitness|fj|fk|flights|florist|flowers|flsmidth|fly|fm|fo|foo|forsale|foundation|fr|frl|frogans|fund|furniture|futbol|ga|gal|gallery|garden|gb|gbiz|gd|ge|gent|gf|gg|ggee|gh|gi|gift|gifts|gives|gl|glass|gle|global|globo|gm|gmail|gmo|gmx|gn|goog|google|gop|gov|gp|gq|gr|graphics|gratis|green|gripe|gs|gt|gu|guide|guitars|guru|gw|gy|hamburg|hangout|haus|healthcare|help|here|hermes|hiphop|hiv|hk|hm|hn|holdings|holiday|homes|horse|host|hosting|house|how|hr|ht|hu|ibm|id|ie|ifm|il|im|immo|immobilien|in|industries|info|ing|ink|institute|insure|int|international|investments|io|iq|ir|irish|is|it|iwc|jcb|je|jetzt|jm|jo|jobs|joburg|jp|juegos|kaufen|kddi|ke|kg|kh|ki|kim|kitchen|kiwi|km|kn|koeln|kp|kr|krd|kred|kw|ky|kyoto|kz|la|lacaixa|land|lat|latrobe|lawyer|lb|lc|lds|lease|legal|lgbt|li|lidl|life|lighting|limited|limo|link|lk|loans|london|lotte|lotto|lr|ls|lt|ltda|lu|luxe|luxury|lv|ly|ma|madrid|maison|management|mango|market|marketing|marriott|mc|md|me|media|meet|melbourne|meme|memorial|menu|mg|mh|miami|mil|mini|mk|ml|mm|mn|mo|mobi|moda|moe|monash|money|mormon|mortgage|moscow|motorcycles|mov|mp|mq|mr|ms|mt|mu|museum|mv|mw|mx|my|mz|na|nagoya|name|navy|nc|ne|net|network|neustar|new|nexus|nf|ng|ngo|nhk|ni|ninja|nl|no|np|nr|nra|nrw|ntt|nu|nyc|nz|okinawa|om|one|ong|onl|ooo|org|organic|osaka|otsuka|ovh|pa|paris|partners|parts|party|pe|pf|pg|ph|pharmacy|photo|photography|photos|physio|pics|pictures|pink|pizza|pk|pl|place|plumbing|pm|pn|pohl|poker|porn|post|pr|praxi|press|pro|prod|productions|prof|properties|property|ps|pt|pub|pw|qa|qpon|quebec|re|realtor|recipes|red|rehab|reise|reisen|reit|ren|rentals|repair|report|republican|rest|restaurant|reviews|rich|rio|rip|ro|rocks|rodeo|rs|rsvp|ru|ruhr|rw|ryukyu|sa|saarland|sale|samsung|sarl|sb|sc|sca|scb|schmidt|schule|schwarz|science|scot|sd|se|services|sew|sexy|sg|sh|shiksha|shoes|shriram|si|singles|sj|sk|sky|sl|sm|sn|social|software|sohu|solar|solutions|soy|space|spiegel|sr|st|style|su|supplies|supply|support|surf|surgery|suzuki|sv|sx|sy|sydney|systems|sz|taipei|tatar|tattoo|tax|tc|td|technology|tel|temasek|tennis|tf|tg|th|tienda|tips|tires|tirol|tj|tk|tl|tm|tn|to|today|tokyo|tools|top|toshiba|town|toys|tp|tr|trade|training|travel|trust|tt|tui|tv|tw|tz|ua|ug|uk|university|uno|uol|us|uy|uz|va|vacations|vc|ve|vegas|ventures|versicherung|vet|vg|vi|viajes|video|villas|vision|vlaanderen|vn|vodka|vote|voting|voto|voyage|vu|wales|wang|watch|webcam|website|wed|wedding|wf|whoswho|wien|wiki|williamhill|wme|work|works|world|ws|wtc|wtf|xyz|yachts|yandex|ye|yoga|yokohama|youtube|yt|za|zm|zone|zuerich|zw))",
            # r'^(?:(?:https?|ftp):\/\/)?(?:\S+(?::\S*)?@)?(?:(?!(?:10|127)(?:\.\d{1,3}){3})(?!(?:169\.254|192\.168)(?:\.\d{1,3}){2})(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})(?:[1-9]\d?|1\d{2}|2[01]\d|22[0-3])(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}(?:\.(?:[1-9]\d?|1\d{2}|2[0-4]\d|25[0-4]))|(?:(?:[a-zA-Z\u00a1-\uffff0-9]+-?)*[a-zA-Z\u00a1-\uffff0-9]+)(?:\.(?:[a-zA-Z\u00a1-\uffff0-9]+-?)*[a-zA-Z\u00a1-\uffff0-9]+)*(?:\.(?:[a-zA-Z\u00a1-\uffff]{2,})))(?::\d{2,5})?(?:\/[^\s]*)?$'
            r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})",
            r"(\d{1,3}\.\d{1,3}\.\[x*\]\.\d{1,3})",
            r"([a-z][_a-z0-9-.]+@[a-z0-9-]+\.[a-z]+)"
            r'^(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2}|:\d{1,5})?$'
        ],
        "MF": [
            # Hash
            r"([a-f0-9]{32}|[A-F0-9]{32})",
            r"([a-f0-9]{40}|[A-F0-9]{40})",
            r"([a-f0-9]{64}|[A-F0-9]{64})",
            # DocFile
            r"([A-Za-z0-9-_\.]*\.(sys|htm|html|js|jpg|png|vb|scr|pif|chm|zip|rar|cab|pdf|doc|docx|ppt|pptx|xls|xlsx|swf|gif|txt|csv))",
            # ExeFile
            r"(?i)([A-Za-z0-9-_\.]*\.(exe|vbs|bat|mdf|mdb|run|apk|app|bin|cgi|wsf|sh|php|py|java|c))",
            # LibFile
            r"(?i)([A-Za-z0-9-_\.]*\.(dll|lib|jar|so|a|pdb))",
            # Registry
            r"((KCU|HKLM|HKCU|HKEY_LOCAL_MACHINE|HKEY_CURRENT_USER|SOFTWARE).{0,1}\\[\\A-Za-z0-9-_]+)",
            r"((HKLM|HKCU|HKEY_LOCAL_MACHINE|HKU)\\\\[\\\\A-Za-z0-9-_]+)",
            r"(HKEY_LOCAL_MACHINE\\|HKLM\\)([a-zA-Z0-9_@\-\^!#.\:\/\$%&+={}\[\]\\*])+"
            # Vulnerability
            r"(CVE\-[0-9]{4}\-[0-9]{4,6})"
        ],
        "SF": [
            # windows-Path
            r'[A-Za-z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*',
            # linux-Path
            r'\/(?:[^\\/:*?"<>|\r\n]+\/)*[^\\/:*?"<>|\r\n]*'
            # SysconfFiles
            r"[A-Za-z]:([\/|\\[Windows|windows]+)\.(ini|xml|config|dat|conf|cnf|sav)",
            r"\/etc\/(crontab|inputrc|syslog|hostname|bashrc|service)",
            r"\/etc([\/[syslog|rsyslog|syslog-ng]+)\.conf",
            r"\/proc\/(cpuinfo|meminfo|mounts)",
            # SyslogFiles
            r"[A-Za-z]:([\/|\\[Windows|windows]+)\.(evt|log|err)",
            r"\/var\/log\/(messages|secure|utmp|wtmp|xferlog|maillog|httpd)",
            r"\/var\/log\/([A-Za-z0-9-_\s]+)\.log",
            # UserconfFiles
            r"[A-Za-z]:([\/|\\[Users]+)\.(ini|xml|config|dat|conf|cnf|sav)",
            r"\/etc\/(passwd|shadow|issue|motd|sudoers)",
            r"\/etc\/ssh\/(sshd_config|ssh_config|ssh_host_dsa_key|ssh_host_rsa_key|ssh_host_key)",
            r"\/etc\/ssh\/(ssh_host_dsa_key|ssh_host_rsa_key|ssh_host_key)\.pub",
            # UserlogFiles
            r"[A-Za-z]:([\/|\\[Users]+)\.(evt|log|err)",
            # AppconfFiles
            r"[A-Za-z]:([\/|\\[Program Files|MySQL|php0-9|apache]+)\.(ini|xml|config|dat|conf|cnf|sav)",
            r"\/etc([\/[my|audisp|audispd|audit|auditd|libaudit|mysql|httpd|conf|resolv|ufw|sysctl|nginx|xinetd]+)\.conf",
            # ApplogFiles
            r"[A-Za-z]:([\/|\\[Program Files|MySQL|php0-9|apache]+)\.(evt|log|err)"
        ],
    }
    for ioc_type, regex_list in IoC_regex.items():
        for regex in regex_list:
            if re.fullmatch(regex, span):
                return True, ioc_type
    return False, None

def batchify(samples, batch_size):
    """
    Batchfy samples with a batch size
    """
    num_samples = len(samples)

    list_samples_batches = []
    
    # if a sentence is too long, make itself a batch to avoid GPU OOM
    to_single_batch = []
    for i in range(0, len(samples)):
        if len(samples[i]['tokens']) > 350:
            to_single_batch.append(i)
    
    for i in to_single_batch:
        logger.info('Single batch sample: %s-%d', samples[i]['doc_key'], samples[i]['sentence_ix'])
        list_samples_batches.append([samples[i]])
    samples = [sample for i, sample in enumerate(samples) if i not in to_single_batch]

    for i in range(0, len(samples), batch_size):
        list_samples_batches.append(samples[i:i+batch_size])

    assert(sum([len(batch) for batch in list_samples_batches]) == num_samples)

    return list_samples_batches

def overlap(s1, s2):
    if s2.start_sent >= s1.start_sent and s2.start_sent <= s1.end_sent:
        return True
    if s2.end_sent >= s1.start_sent and s2.end_sent <= s1.end_sent:
        return True
    return False


def convert_dataset_to_samples(dataset, max_span_length, ner_label2id=None, context_window=0, split=0):
    """
    Extract sentences and gold entities from a dataset
    """
    # split: split the data into train and dev (for ACE04)
    # split == 0: don't split
    # split == 1: return first 90% (train)
    # split == 2: return last 10% (dev)
    samples = []
    num_ner = 0
    max_len = 0
    max_ner = 0
    num_overlap = 0

    if split == 0:
        data_range = (0, len(dataset))
    elif split == 1:
        data_range = (0, int(len(dataset) * 0.9))
    elif split == 2:
        data_range = (int(len(dataset) * 0.9), len(dataset))

    for c, doc in enumerate(dataset):
        if c < data_range[0] or c >= data_range[1]:
            continue
        for i, sent in enumerate(doc):
            num_ner += len(sent.ner)
            sample = {
                'doc_key': doc._doc_key,
                'sentence_ix': sent.sentence_ix,
            }
            if context_window != 0 and len(sent.text) > context_window:
                logger.info('Long sentence: {} {}'.format(sample, len(sent.text)))

            sample['tokens'] = sent.text
            sample['sent_length'] = len(sent.text)
            sent_start = 0
            sent_end = len(sample['tokens'])

            max_len = max(max_len, len(sent.text))
            max_ner = max(max_ner, len(sent.ner))

            if context_window > 0:
                add_left = (context_window - len(sent.text)) // 2
                add_right = (context_window - len(sent.text)) - add_left

                # add left context
                j = i - 1
                while j >= 0 and add_left > 0:
                    context_to_add = doc[j].text[-add_left:]
                    sample['tokens'] = context_to_add + sample['tokens']
                    add_left -= len(context_to_add)
                    sent_start += len(context_to_add)
                    sent_end += len(context_to_add)
                    j -= 1

                j = i + 1
                while j < len(doc) and add_right > 0:
                    context_to_add = doc[j].text[:add_right]
                    sample['tokens'] = sample['tokens'] + context_to_add
                    add_right -= len(context_to_add)
                    j += 1

            sample['sent_start'] = sent_start
            sample['sent_end'] = sent_end
            sample['sent_start_in_doc'] = sent.sentence_start

            sent_ner = {}
            for ner in sent.ner:
                sent_ner[ner.span.span_sent] = ner.label

            '''for ner, label in sent_ner.items():
                ner_list = []
                for n in range(ner[0], ner[1] + 1):
                    ner_list.append(n)
                sent_ner_list[tuple(ner_list)] = label'''


            span2id = {}
            sample['spans'] = []
            sample['spans_label'] = []
            for i in range(len(sent.text)):
                for j in range(i, min(len(sent.text), i + max_span_length)):
                    sample['spans'].append((i + sent_start, j + sent_start, j - i + 1))
                    span2id[(i, j)] = len(sample['spans']) - 1

                    '''span_list = []
                    for s in range(i, j+1):
                        span_list.append(s)
                    for ner, label in sent_ner_list.items():
                        ner_list = []
                        for n in ner:
                            ner_list.append(n)
                        if list(set(span_list)&set(ner_list)):
                            sample['spans_label'].append(ner_label2id[label])
                        else:
                            sample['spans_label'].append(0)'''

                    if (i, j) not in sent_ner:
                        sample['spans_label'].append(0)
                    else:
                        sample['spans_label'].append(ner_label2id[sent_ner[(i, j)]])

            samples.append(sample)
    avg_length = sum([len(sample['tokens']) for sample in samples]) / len(samples)
    max_length = max([len(sample['tokens']) for sample in samples])
    logger.info('# Overlap: %d' % num_overlap)
    logger.info('Extracted %d samples from %d documents, with %d NER labels, %.3f avg input length, %d max length' % (
    len(samples), data_range[1] - data_range[0], num_ner, avg_length, max_length))
    logger.info('Max Length: %d, max NER: %d' % (max_len, max_ner))
    return samples, num_ner


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def get_train_fold(data, fold):
    print('Getting train fold %d...'%fold)
    l = int(len(data) * 0.1 * fold)
    r = int(len(data) * 0.1 * (fold+1))
    new_js = []
    new_docs = []
    for i in range(len(data)):
        if i < l or i >= r:
            new_js.append(data.js[i])
            new_docs.append(data.documents[i])
    print('# documents: %d --> %d'%(len(data), len(new_docs)))
    data.js = new_js
    data.documents = new_docs
    return data

def get_test_fold(data, fold):
    print('Getting test fold %d...'%fold)
    l = int(len(data) * 0.1 * fold)
    r = int(len(data) * 0.1 * (fold+1))
    new_js = []
    new_docs = []
    for i in range(len(data)):
        if i >= l and i < r:
            new_js.append(data.js[i])
            new_docs.append(data.documents[i])
    print('# documents: %d --> %d'%(len(data), len(new_docs)))
    data.js = new_js
    data.documents = new_docs
    return data
