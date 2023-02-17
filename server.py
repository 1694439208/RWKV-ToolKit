import streamlit as st




	
tab1, tab2,tab3,tab4= st.tabs(["训练", "识别","转化模型","Chat-RWKV3"])


with tab1:
    ctx_len = st.number_input('ctx_len', min_value=0, max_value=4096, value=1024, step=1,key =1)
    n_layer = st.number_input('n_layer', min_value=0, max_value=100, value=6, step=1,key =2)
    n_embd = st.number_input('n_embd', min_value=0, max_value=4096, value=512, step=1,key =3)
    batch_size = st.number_input('batch_size', min_value=0, max_value=100, value=2, step=1,key =4)
    epoch_save_frequency = st.number_input('保存间隔', min_value=0, max_value=100, value=2, step=1,key =5)
    epoch_length_fixed = st.number_input('epoch_length_fixed', min_value=0, max_value=999999, value=10000, step=1,key =6)
    n_epoch = st.number_input('n_epoch', min_value=0, max_value=99999, value=500, step=1,key =7)
    datafile = st.text_input('输入数据集路径 utf8编码',key =8)
    vocab_size1 = None
    option = None
    Trin_model = False
    if st.checkbox("断点训练"):
        import os
        option = st.selectbox(
        '选择一个模型',
        [fn for fn in os.listdir("..\\") if fn.endswith("pth")],key=11)
        if st.checkbox("微调新语料训练"):
            Trin_model = True
            #vocab_size1 = st.number_input('微调尺寸', min_value=0, max_value=999999, value=5000, step=1,key =17)

    

    if st.button('开始训练',key =10):
        
        ########################################################################################################
        # The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
        ########################################################################################################

        import os

        # if False: # True False ---> Set to False if you don't understand it
        #     print("\n\n[[[ SPECIAL DEBUG MODE FOR MYSELF. DON'T ENABLE THIS IF YOU DON'T UNDERSTAND IT ]]]\n\n")
        #     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        #     import src.utils
        #     src.utils.set_seed(42) # make training deterministic (including dataloader). if you are doing this, remember to change seed when you load a model (otherwise the dataloader loads old samples)

        import logging
        import datetime
        from src.model import GPT, GPTConfig
        from src.trainer import Trainer, TrainerConfig
        from src.utils import Dataset
        import torch
        import numpy as np

        np.set_printoptions(precision=4, suppress=True, linewidth=200)
        logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO,)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        ### Step 1: set training data ##########################################################################
        datafile_encoding = 'utf-8'
        model_type = 'RWKV'
        lr_init = 8e-4 # we can use larger lr because of preLN
        lr_final = 1e-5

        # the mini-epoch is very short and of fixed length (length = ctx_len * epoch_length_fixed tokens)
        #n_epoch = 500
        #epoch_length_fixed = 10000

        # 0 = never, 1 = every mini-epoch, 2 = every two mini-epochs, ...
        epoch_save_path = '..\\trained-'

        ########################################################################################################

        grad_norm_clip = 1.0
        warmup_tokens = ctx_len * batch_size * 0

        betas = (0.9, 0.99)
        eps = 4e-9

        num_workers = 0

        ########################################################################################################
        # Load data
        ########################################################################################################

        print('loading data... ' + datafile)
        train_dataset = Dataset(open(
            datafile, "r", encoding=datafile_encoding).read(), ctx_len, epoch_length_fixed,Trin_model)


        if Trin_model:
            MODEL_NAME = "..\\" + os.path.splitext(option)[0]
            m2 = torch.load(MODEL_NAME+".pth")
            train_dataset.vocab_size = m2["emb.weight"].shape[0]
            print(f"网络对齐：{train_dataset.vocab_size}")
        ########################################################################################################
        # Train model
        ########################################################################################################
            

        model = GPT(GPTConfig(train_dataset.vocab_size, train_dataset.ctx_len, model_type=model_type,
                                n_layer=n_layer, n_embd=n_embd)).cuda()

        ### ---> load a trained model <---
        if option != None:
            MODEL_NAME = "..\\" + os.path.splitext(option)[0]
            m2 = torch.load(MODEL_NAME+".pth")
            print(train_dataset.vocab_size,m2["emb.weight"].shape,train_dataset.vocab_size-m2["emb.weight"].shape[0])#head.weight
            if train_dataset.vocab_size > m2["emb.weight"].shape[0]:
                pad = torch.nn.ZeroPad2d(padding=(0,0,0,train_dataset.vocab_size-m2["emb.weight"].shape[0]))
                m2["emb.weight"] = pad(m2["emb.weight"])
                m2["head.weight"] = pad(m2["head.weight"])
            print(train_dataset.vocab_size,m2["emb.weight"].shape)#head.weight
            if "intput.parameter" in m2:
                del m2["intput.parameter"]
            
            model.load_state_dict(m2)
        
        print('model', model_type, 'epoch', n_epoch, 'batchsz', batch_size, 'betas',
            betas, 'eps', eps, 'ctx', ctx_len, 'layer', n_layer, 'embd', n_embd, )
        tconf = TrainerConfig(model_type=model_type, max_epochs=n_epoch, batch_size=batch_size,
                            learning_rate=lr_init, lr_decay=True, lr_final=lr_final, betas=betas, eps=eps, grad_norm_clip=grad_norm_clip,
                            warmup_tokens=warmup_tokens, final_tokens=n_epoch*len(train_dataset)*ctx_len, num_workers=num_workers, epoch_save_frequency=epoch_save_frequency, epoch_save_path=epoch_save_path)
        trainer = Trainer(model, train_dataset, None, tconf)

        trainer.train()
        #self.config.epoch_save_path + str(epoch+1) + '.pth'
        torch.save(model.state_dict(), epoch_save_path + str(n_epoch+1) + '-' + trainer.get_run_name() +
                '-' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S') + '.pth')


with tab2:
    UNKNOWN_CHAR  = st.text_input('输入空字符-默认不要动，出错再改'," ",key=15)
    import os
    option = st.selectbox(
     '选择一个模型',
    [fn for fn in os.listdir("..\\") if fn.endswith("pth")])
    txt = st.text_area('输入提示文本',"当他拿起枪")
    genre = st.radio(
    "选择运行时架构",
    ('cpu', 'cuda'))
    NUM_TRIALS = st.number_input('循环次数', min_value=0, max_value=100, value=2, step=1)
    LENGTH_PER_TRIAL = st.number_input('生成字数', min_value=0, max_value=1000, value=500, step=1)
    if st.button('开始生成'):
        import numpy as np
        import math
        import time
        import types
        import copy
        import torch
        from torch.nn import functional as F
        from src.utils import TOKENIZER, Dataset
        from src.model_run import RWKV_RNN
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        np.set_printoptions(precision=4, suppress=True, linewidth=200)

        ### Step 1: set model ##################################################################################
        model_type = 'RWKV'           # 'RWKV' or 'RWKV-ffnPre'

        # your trained model
        MODEL_NAME = "..\\" + os.path.splitext(option)[0]
        WORD_NAME = '..\\vocab'           # the .json vocab (generated by train.py

        # --> set UNKNOWN_CHAR to the rarest token in your vocab.json <--
        # --> all unknown tokens in your context will be denoted by it <--
        #UNKNOWN_CHAR = ' '   # here we just set it to [space] for simplicity

        RUN_DEVICE = genre   # 'cpu' (already very fast) or 'cuda'
        DEBUG_DEBUG = False  # True False - show softmax output

        ### Step 2: set context ################################################################################

        context = txt       # ==> this is your prompt

        TEMPERATURE = 1.0
        top_p = 0.5
        top_p_newline = 0.9

        ########################################################################################################

        print(f'Loading {MODEL_NAME}...')
        model = RWKV_RNN(MODEL_NAME, RUN_DEVICE, model_type, 0, 0, 0)
        tokenizer = TOKENIZER(WORD_NAME, UNKNOWN_CHAR=UNKNOWN_CHAR)

        ########################################################################################################

        context = tokenizer.refine_context(context)
        print('\nYour prompt has ' + str(len(context)) + ' tokens.')
        print('\n--> Currently the first run takes a while if your prompt is long, as we are using RNN to process the prompt. Use GPT to build the hidden state for better speed. <--\n')
        

        for TRIAL in range(1 if DEBUG_DEBUG else NUM_TRIALS):
            t_begin = time.time_ns()

            src_len = len(context)
            ctx = [tokenizer.stoi.get(s, tokenizer.UNKNOWN_CHAR) for s in context]
            st.header("----------------------------" + context)

            model.clear()
            if TRIAL == 0:
                init_state = types.SimpleNamespace()
                for i in range(src_len):
                    x = ctx[:i+1]
                    if i == src_len - 1:
                        init_state.out = model.run(x)
                    else:
                        model.run(x)
                model.save(init_state)
            else:
                model.load(init_state)
            outtext = f"**{txt}**"
            t = st.empty()
            for i in range(src_len, src_len + (1 if DEBUG_DEBUG else LENGTH_PER_TRIAL)):
                x = ctx[:i+1]
                x = x[-ctx_len:]

                if i == src_len:
                    out = copy.deepcopy(init_state.out)
                else:
                    out = model.run(x)
                if DEBUG_DEBUG:
                    print('model', np.array(x), '==>', np.array(
                        out), np.max(out), np.min(out))

                char = tokenizer.sample_logits(out, x, ctx_len, temperature=TEMPERATURE,
                                            top_p_usual=top_p, top_p_newline=top_p_newline)
                char = char.item()
                outtext += tokenizer.itos[int(char)]
                t.markdown("%s" % outtext)
                ctx += [char]
            t_end = time.time_ns()


with tab3:
    model_type = 'RWKV'
    RUN_DEVICE = 'cpu'
    
    import torch
    from torch.nn import functional as F
    from src.model_run import RWKV_RNN_ONNX
    st.text('此功能仅为了导出到第三方开发工具推理，不需要的可以不看，第三方开发工具还没优化完')
    option = st.selectbox(
     '选择一个模型生成onnx',
    [fn for fn in os.listdir("..\\") if fn.endswith("pth")],key=21)
    if st.button('开始生成',key=22):
        MODEL_NAME = "..\\" + os.path.splitext(option)[0]
        model = RWKV_RNN_ONNX(MODEL_NAME, RUN_DEVICE, model_type, 0, 0, 0)#8449

        ctx_len = int(model.ctx_len)
        n_embd = int(model.n_embd)
        n_layer = int(model.n_layer)
        xx_att = torch.zeros(model.n_layer, model.n_embd)
        aa_att = torch.zeros(model.n_layer, model.n_embd)
        bb_att = torch.zeros(model.n_layer, model.n_embd)
        xx_ffn = torch.zeros(model.n_layer, model.n_embd)

        ctx = torch.randint(ctx_len, (ctx_len,), dtype=torch.int32 ) + 100

        torch.onnx.export(model, 
            args=(ctx, xx_att, aa_att, bb_att, xx_ffn),
            f=f"..\\v3.onnx", 
            input_names = ["idx", "xx_att", "aa_att", "bb_att", "xx_ffn"], 
            output_names = ["x", "xx_att_r", "aa_att_r", "bb_att_r", "xx_ffn_r"],
            opset_version=11, # 使用版本 10
            verbose=True)
        
        binfile = open(f"..\\v3.onnx", 'rb')
        resfile = open(f'..\\base.dat', 'wb')
        size = os.path.getsize(f"..\\v3.onnx")
        resfile.write(ctx_len.to_bytes(4, 'little'))
        resfile.write(n_layer.to_bytes(4, 'little'))
        resfile.write(n_embd.to_bytes(4, 'little'))
        data = binfile.read(size)
        resfile.write(data)
        binfile.close()
        resfile.close()
        st.text("生成成功:base.dat")
        print("生成成功")
with tab4:
    user = "User"
    bot = "Bot"
    interface = ":"
    st.text('此功能是从作者chat-rwkv4neo移植到3的，仅为个人学习使用')
    UNKNOWN_CHAR  = st.text_input('输入空字符-默认不要动，出错再改'," ",key=35)
    import os
    option = st.selectbox(
     '选择一个模型',
    [fn for fn in os.listdir("..\\") if fn.endswith("pth")],key=36)
    txt = st.text_area('输入提示文本',"+gen ")
    genre = st.radio(
    "选择运行时架构",
    ('cpu', 'cuda'),key=37)
    LENGTH_PER_TRIAL = st.number_input('生成字数', min_value=0, max_value=1000, value=500, step=1,key=39)

    
    if st.button('开始生成',key=41):
        import numpy as np
        import math
        import time
        import types
        import copy
        import torch
        from torch.nn import functional as F
        from src.utils import TOKENIZER, Dataset
        from src.model_run import RWKV_RNN
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        np.set_printoptions(precision=4, suppress=True, linewidth=200)

        ### Step 1: set model ##################################################################################
        model_type = 'RWKV'           # 'RWKV' or 'RWKV-ffnPre'

        # your trained model
        MODEL_NAME = "..\\" + os.path.splitext(option)[0]
        WORD_NAME = '..\\vocab'           # the .json vocab (generated by train.py

        # --> set UNKNOWN_CHAR to the rarest token in your vocab.json <--
        # --> all unknown tokens in your context will be denoted by it <--
        #UNKNOWN_CHAR = ' '   # here we just set it to [space] for simplicity

        RUN_DEVICE = genre   # 'cpu' (already very fast) or 'cuda'
        DEBUG_DEBUG = False  # True False - show softmax output

        ### Step 2: set context ################################################################################

        context = txt       # ==> this is your prompt

        TEMPERATURE = 1.0
        top_p = 0.5
        top_p_newline = 0.9

        ########################################################################################################

        print(f'Loading {MODEL_NAME}...')
        model = RWKV_RNN(MODEL_NAME, RUN_DEVICE, model_type, 0, 0, 0)
        tokenizer = TOKENIZER(WORD_NAME, UNKNOWN_CHAR=UNKNOWN_CHAR)


        CHAT_LEN_SHORT = 40 #应该是上下文长度
        CHAT_LEN_LONG = 150
        FREE_GEN_LEN = 200 #生成长度

        GEN_TEMP = 1.0
        GEN_TOP_P = 0.85  

        AVOID_REPEAT = '，。：？！'

        model_tokens = []
        model_state = None

        AVOID_REPEAT_TOKENS = []
        for i in AVOID_REPEAT:
            dd = tokenizer.encode(i)
            assert len(dd) == 1
            AVOID_REPEAT_TOKENS += dd
        def run_rnn(tokens, newline_adj = 0):
            global model_tokens, model_state

            tokens = [int(x) for x in tokens]
            model_tokens += tokens

            context = tokens
            src_len = len(context)
            ctx = [tokenizer.stoi.get(s, tokenizer.UNKNOWN_CHAR) for s in context]
            model.clear()


            out, model_state = model.forward(tokens, model_state,tokenizer)

            # print(f'### model ###\n{tokens}\n[{tokenizer.decode(model_tokens)}]')

            out[0] = -999999999  # disable <|endoftext|>
            out[187] += newline_adj # adjust \n probability
            # if newline_adj > 0:
            #     out[15] += newline_adj / 2 # '.'
            if model_tokens[-1] in AVOID_REPEAT_TOKENS:
                out[model_tokens[-1]] = -999999999
            return out
        all_state = {}
        def save_all_stat(srv, name, last_out):
            n = f'{name}_{srv}'
            all_state[n] = {}
            all_state[n]['out'] = last_out
            all_state[n]['rnn'] = copy.deepcopy(model_state)
            all_state[n]['token'] = copy.deepcopy(model_tokens)

        def load_all_stat(srv, name):
            global model_tokens, model_state
            n = f'{name}_{srv}'
            model_state = copy.deepcopy(all_state[n]['rnn'])
            model_tokens = copy.deepcopy(all_state[n]['token'])
            return all_state[n]['out']
        ctx = [tokenizer.stoi.get(s, tokenizer.UNKNOWN_CHAR) for s in f'aaa']
        out = run_rnn(ctx)
        st.text(f"生成成功:base.dat:{out}")
        save_all_stat('', 'chat_init', out)
        torch.cuda.empty_cache()
        srv_list = ['dummy_server']
        for s in srv_list:
            save_all_stat(s, 'chat', out)

        st.text(f'### prompt ###\n[{tokenizer.decode(model_tokens)}]\n')
        def reply_msg(msg):
            st.text(f'{bot}{interface} {msg}\n')

        def on_message(message):
            global model_tokens, model_state

            srv = 'dummy_server'

            msg = message.replace('\\n','\n').strip()
            # if len(msg) > 1000:
            #     reply_msg('your message is too long (max 1000 tokens)')
            #     return

            x_temp = GEN_TEMP
            x_top_p = GEN_TOP_P
            if ("-temp=" in msg):
                x_temp = float(msg.split("-temp=")[1].split(" ")[0])
                msg = msg.replace("-temp="+f'{x_temp:g}', "")
                # print(f"temp: {x_temp}")
            if ("-top_p=" in msg):
                x_top_p = float(msg.split("-top_p=")[1].split(" ")[0])
                msg = msg.replace("-top_p="+f'{x_top_p:g}', "")
                # print(f"top_p: {x_top_p}")
            if x_temp <= 0.2:
                x_temp = 0.2
            if x_temp >= 5:
                x_temp = 5
            if x_top_p <= 0:
                x_top_p = 0
            
            if msg == '+reset':
                out = load_all_stat('', 'chat_init')
                save_all_stat(srv, 'chat', out)
                reply_msg("Chat reset.")
                return

            elif msg[:5].lower() == '+gen ' or msg[:4].lower() == '+qa ' or msg[:4].lower() == '+qq ' or msg.lower() == '+++' or msg.lower() == '++':

                if msg[:5].lower() == '+gen ':
                    new = '\n' + msg[5:].strip()
                    # print(f'### prompt ###\n[{new}]')
                    model_state = None
                    model_tokens = []
                    out = run_rnn(tokenizer.encode(new))
                    save_all_stat(srv, 'gen_0', out)

                elif msg[:4].lower() == '+qq ':
                    new = '\nQ: ' + msg[4:].strip() + '\nA:'
                    # print(f'### prompt ###\n[{new}]')
                    model_state = None
                    model_tokens = []
                    out = run_rnn(tokenizer.encode(new))
                    save_all_stat(srv, 'gen_0', out)

                elif msg[:4].lower() == '+qa ':
                    out = load_all_stat('', 'chat_init')

                    real_msg = msg[4:].strip()
                    new = f"{user}{interface} {real_msg}\n\n{bot}{interface}"
                    # print(f'### qa ###\n[{new}]')
                    
                    out = run_rnn(tokenizer.encode(new))
                    save_all_stat(srv, 'gen_0', out)

                elif msg.lower() == '+++':
                    try:
                        out = load_all_stat(srv, 'gen_1')
                        save_all_stat(srv, 'gen_0', out)
                    except:
                        return

                elif msg.lower() == '++':
                    try:
                        out = load_all_stat(srv, 'gen_0')
                    except:
                        return

                begin = len(model_tokens)
                out_last = begin
                t = st.empty()
                outtext = f"****"
                for i in range(FREE_GEN_LEN+100):
                    token = tokenizer.sample_logits1(
                        out,
                        model_tokens,
                        ctx_len,
                        temperature=x_temp,
                        top_p=x_top_p,
                    )
                    if msg[:4].lower() == '+qa ':# or msg[:4].lower() == '+qq ':
                        out = run_rnn([token], newline_adj=-2)
                    else:
                        out = run_rnn([token])
                    
                    xxx = tokenizer.decode(model_tokens[out_last:])
                    if '\ufffd' not in xxx: # avoid utf-8 display issues
                        outtext+=xxx
                        t.markdown("%s" % outtext)
                        out_last = begin + i + 1
                        if i >= FREE_GEN_LEN:
                            break
                print('\n')
                # send_msg = tokenizer.decode(model_tokens[begin:]).strip()
                # print(f'### send ###\n[{send_msg}]')
                # reply_msg(send_msg)
                save_all_stat(srv, 'gen_1', out)

            else:
                if msg.lower() == '+':
                    try:
                        out = load_all_stat(srv, 'chat_pre')
                    except:
                        return
                else:
                    out = load_all_stat(srv, 'chat')
                    new = f"{user}{interface} {msg}\n\n{bot}{interface}"
                    # print(f'### add ###\n[{new}]')
                    out = run_rnn(tokenizer.encode(new), newline_adj=-999999999)
                    save_all_stat(srv, 'chat_pre', out)

                begin = len(model_tokens)
                out_last = begin
                st.text(f'{bot}{interface}\n')
                t = st.empty()
                outtext = f"****"
                for i in range(LENGTH_PER_TRIAL):
                    if i <= 0:
                        newline_adj = -999999999
                    elif i <= CHAT_LEN_SHORT:
                        newline_adj = (i - CHAT_LEN_SHORT) / 10
                    elif i <= CHAT_LEN_LONG:
                        newline_adj = 0
                    else:
                        newline_adj = (i - CHAT_LEN_LONG) * 0.25 # MUST END THE GENERATION
                    token = tokenizer.sample_logits1(
                        out,
                        model_tokens,
                        ctx_len,
                        temperature=x_temp,
                        top_p=x_top_p,
                    )
                    out = run_rnn([token], newline_adj=newline_adj)

                    xxx = tokenizer.decode(model_tokens[out_last:])
                    if '\ufffd' not in xxx: # avoid utf-8 display issues
                        outtext+=xxx[0]
                        t.markdown("%s" % outtext)
                        out_last = begin + i + 1
                    
                    send_msg = tokenizer.decode(model_tokens[begin:])
                    if '\n\n' in send_msg:
                        send_msg = send_msg.strip()
                        break
                    
                    # send_msg = tokenizer.decode(model_tokens[begin:]).strip()
                    # if send_msg.endswith(f'{user}{interface}'): # warning: needs to fix state too !!!
                    #     send_msg = send_msg[:-len(f'{user}{interface}')].strip()
                    #     break
                    # if send_msg.endswith(f'{bot}{interface}'):
                    #     send_msg = send_msg[:-len(f'{bot}{interface}')].strip()
                    #     break

                # print(f'{model_tokens}')
                # print(f'[{tokenizer.decode(model_tokens)}]')

                # print(f'### send ###\n[{send_msg}]')
                # reply_msg(send_msg)
                save_all_stat(srv, 'chat', out)

        print(f'''指令:
直接输入内容 --> 和机器人聊天（建议问机器人问题），用\\n代表换行
+ --> 让机器人换个回答
+reset --> 重置对话
+gen 某某内容 --> 续写任何中英文内容，用\\n代表换行
+qa 某某问题 --> 问独立的问题（忽略上下文），用\\n代表换行
+qq 某某问题 --> 问独立的问题（忽略上下文），且敞开想象力，用\\n代表换行
+++ --> 继续 +gen / +qa / +qq 的回答
++ --> 换个 +gen / +qa / +qq 的回答
作者：彭博 请关注我的知乎: https://zhuanlan.zhihu.com/p/603840957
如果喜欢，欢迎看我们的优质护眼灯: https://withablink.taobao.com
现在可以输入内容和机器人聊天（注意它不大懂中文，它更懂英文）。请经常使用 +reset 重置机器人记忆。
目前没有“重复惩罚”，所以机器人有时会重复，此时必须使用 + 换成正常回答，以免污染电脑记忆。
注意：和上下文无关的独立问题，必须用 +qa 或 +qq 问，以免污染电脑记忆。
请先试下列咒语，理解咒语的写法！咒语至关重要。
+gen \\n活动出席发言稿：\\n大家好，
+gen \\n怎样创立一家快速盈利的AI公司：\\n1.
+gen 二向箔是一种超级武器，它的原理是
+gen \\nimport torch
【下面这些多试几次】
+qq 请以《我的驴》为题写一篇作文
+qq 请以《企鹅》为题写一首诗歌
+qq 请设定一个奇幻世界，告诉我详细的世界设定。
''')
        #print(f'Ready - {CHAT_LANG} {args.RUN_DEVICE} {args.FLOAT_MODE} QA_PROMPT={QA_PROMPT} {args.MODEL_NAME}\n')

        msg = f'{user}{interface} ' + txt
        if len(msg.strip()) > 0:
            on_message(msg)
        else:
            st.text('Error: please say something')

        st.text("生成成功")
