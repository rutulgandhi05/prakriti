import os
import yaml
import torch
import random
import imageio
import argparse
import torchvision.transforms as T

from tqdm import auto
from PIL import Image, ImageEnhance
from diffusers import DiffusionPipeline, DDPMScheduler


class Teacher:
    def __init__(self, args_path):
        self.args = self.config_2_args(args_path)

        # INPUT images
        self.imgs_path=os.path.join(self.args.train_data_dir, self.args.character_name, "0")
        self.imgs_wh=(1024,1024) # 25 min for 500 steps (3090TI) -> noisy when used with lower INPUT image resolution
        #imgs_wh=(768,768) # 15 min for 500 steps (3090TI) -> good results
        #imgs_wh=(512,512) # 10 min for 500 steps (3090TI) -> fastest
        self.imgs_flip=True # additionally use horizontally mirrored INPUT images

        self.learn_token=self.args.placeholder_token
        self.start_token=self.args.initializer_token
        self.learning_rates=[(4,1e-3),(8,9e-4),(13,8e-4),(20,7e-4),(35,6e-4),(60,5e-4),(100,4e-4),(160,3e-4)]

        # OUTPUT embedding
        self.embs_path = self.args.teacher_output_dir
        self.emb_file = f"{self.args.character_name}.pt"

        # Visualize intermediate optimization steps
        self.test_prompt = "a photo of {} at the beach"
        self.intermediate_steps=9
        self.outGIF = f"{self.args.teacher_backup_data_dir_root}/{self.args.character_name}_train.gif"

        # SDXL base model
        self.base = self.base_model(self.args.pretrained_model_name_or_path)

        self.template_prompts_for_faces=["a color photo of {}",
                            "a national geograhic photo of {}",
                            "a national geograhic shot of {}",
                            "a shot of {}",
                            "a studio shot of {}", 
                            "a selfie of {}",
                            "a SLR photo of {}",
                            "a photo of {}",
                            "a studio photo of {}",
                            "a cropped photo of {}",
                            "a close-up photo of {}",
                            "an award winning photo of {}",
                            "a good photo of {}",
                            "a portrait photo of {}",
                            "a portrait shot of {}",
                            "a SLR photo of a cool {}",
                            "a SLR photo of the face of {}",
                            "a funny portrait of {}",
                            "{}, portrait shot",
                            "{}, studio lighting",
                            "{}, bokeh",
                            "{}, professional photo"]

        self.prompts=self.template_prompts_for_faces

        self.negative_prompt='''
                        deformed, ugly, disfigured, blurry, pixelated, hideous, 
                        indistinct, old, malformed, extra hands, extra arms, joined misshapen, collage, grainy, 
                        low, poor, monochrome, huge, extra fingers, mutated hands, cropped off, out of frame,
                        poorly drawn hands, mutated hands, fused fingers, too many fingers, fused finger, closed eyes,
                        cropped face, blur, long body, people, watermark, text, logo, signature, text, logo, writing,
                        heading, no text, logo, wordmark, writing, heading, signature, 2 heads, 2 faces, b&w, nude, naked
                        '''

        self.prompt_variations=[", wearing white t-shirt, white background"]

    
    def config_2_args(self, path):
        with open(path, 'r') as file:
            yaml_data = yaml.safe_load(file)
        parser = argparse.ArgumentParser(description="Generate args from config")
        for key, value in yaml_data.items():
            parser.add_argument(f'--{key}', type=type(value), default=value)
        
        args = parser.parse_args([])
            
        return args
    
    def base_model(self, base_model_path):
        base = DiffusionPipeline.from_pretrained(
            base_model_path, 
            torch_dtype=torch.bfloat16,
            variant="fp32", 
            use_safetensors=False,
            add_watermarker=False,
            # DDPM DDPMScheduler instead of default EulerDiscreteScheduler 
            scheduler = DDPMScheduler(num_train_timesteps=1000,prediction_type="epsilon",beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False)
        )
        base.disable_xformers_memory_efficient_attention()
        torch.set_grad_enabled(True)
        base.to("cuda")

        return base
    
    def force_training_grad(self, model, bT = True, bG = True):
        model.training = bT
        model.requires_grad_ = bG
        for module in model.children():
            self.force_training_grad(module,bT,bG)

    def load_imgs(self, path, wh=(1024,1024), flip=True, preview=(64,64)):
        files = list()
        imgs = list()
        PILimgs = list()
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in [f for f in filenames if (f.endswith(".jpg") or f.endswith(".JPG") or f.endswith(".png") or f.endswith(".JPEG") or f.endswith(".jpeg"))]:
                fname = os.path.join(dirpath, filename)
                files.append(fname)
        for f in files:
            img = Image.open(f).convert("RGB")
            img = T.RandomAutocontrast(p=1.0)(img)
            img = T.Resize(wh, interpolation=T.InterpolationMode.LANCZOS)(img)
            #img = ImageEnhance.Contrast(T.RandomAutocontrast(p=1.0)(img)).enhance(5.0)
            PILimgs.append(T.Resize(preview, interpolation=T.InterpolationMode.LANCZOS)(img))
            img0 = T.ToTensor()(img)
            img0 = img0 *2.- 1.0
            imgs.append(img0[None].clip(-1.,1.))
            # plus horizontally mirrowed
            if flip:
                img0 = T.RandomHorizontalFlip(p=1.0)(img0)  
                imgs.append(img0[None].clip(-1.,1.)) 
                img = T.RandomHorizontalFlip(p=1.0)(img)
                PILimgs.append(T.Resize(preview, interpolation=T.InterpolationMode.LANCZOS)(img))
        return imgs, PILimgs
    

    def make_grid(self, imgs):
        n=len(imgs)
        cols=1
        while cols*cols<n:
            cols+=1
        rows=n//cols+int(n%cols>0)
        w, h = imgs[0].size
        grid = Image.new('RGB', size=(cols*w, rows*h))  
        for i, img in enumerate(imgs):
            grid.paste(img, box=(i%cols*w, i//cols*h))
        return grid
    
    
    def save_XLembedding(self, emb, embedding_file, path):
        torch.save(emb, path+embedding_file)

    
    def set_XLembedding(self, base, emb, token):
        with torch.no_grad():            
            # Embeddings[tokenNo] to learn
            tokens=base.components["tokenizer"].encode(token)
            assert len(tokens)==3, "token is not a single token in 'tokenizer'"
            tokenNo=tokens[1]
            tokens=base.components["tokenizer_2"].encode(token)
            assert len(tokens)==3, "token is not a single token in 'tokenizer_2'"
            tokenNo2=tokens[1]
            embs=base.components["text_encoder"].text_model.embeddings.token_embedding.weight
            embs2=base.components["text_encoder_2"].text_model.embeddings.token_embedding.weight
            assert embs[tokenNo].shape==emb["emb"].shape, "different 'text_encoder'"
            assert embs2[tokenNo2].shape==emb["emb2"].shape, "different 'text_encoder_2'"
            embs[tokenNo]=emb["emb"].to(embs.dtype).to(embs.device)
            embs2[tokenNo2]=emb["emb2"].to(embs2.dtype).to(embs2.device)


    def load_XLembedding(self, base, token, embedding_file, path):
        emb=torch.load(path+embedding_file)
        self.set_XLembedding(base, emb, token)


    def XL_textual_inversion(self, base, imgs, prompts, 
                             prompt_variations = None, 
                             token="my", start_token = None,
                             negative_prompt=None, 
                             learning_rates=[(5,1e-3),(10,9e-4),(20,8e-4),(35,7e-4),(55,6e-4),(80,5e-4),(110,4e-4),(145,3e-4)], 
                             intermediate_steps=9):
    
        XLt1=base.components["text_encoder"]
        XLt2=base.components["text_encoder_2"]
        XLtok1=base.components["tokenizer"]
        XLtok2=base.components["tokenizer_2"]
        XLunet=base.components["unet"]
        XLvae=base.components['vae']
        XLsch=base.components['scheduler']
        base.upcast_vae() # vae does not work correctly in 16 bit mode -> force fp32
        
        # Check Scheduler
        schedulerType=XLsch.config.prediction_type
        assert schedulerType in ["epsilon","sample"], "{} scheduler not supported".format(schedulerType)

        # Embeddings to Finetune
        embs=XLt1.text_model.embeddings.token_embedding.weight
        embs2=XLt2.text_model.embeddings.token_embedding.weight

        with torch.no_grad():       
            # Embeddings[tokenNo] to learn
            tokens=XLtok1.encode(token)
            assert len(tokens)==3, "token is not a single token in 'tokenizer'"
            tokenNo=tokens[1]
            tokens=XLtok2.encode(token)
            assert len(tokens)==3, "token is not a single token in 'tokenizer_2'"
            tokenNo2=tokens[1]            

            # init Embedding[tokenNo] with noise or with a copy of an existing embedding
            if start_token=="man" or start_token==None:
                # Original value range: [-0.5059,0.6538] # regular [-0.05,+0.05]
                embs[tokenNo]=(torch.randn_like(embs[tokenNo])*.01).clone() # start with [-0.04,+0.04]
                # Original value range 2: [-0.6885,0.1948] # regular [-0.05,+0.05]
                embs2[tokenNo2]=(torch.randn_like(embs2[tokenNo2])*.01).clone() # start [-0.04,+0.04]
                startNo="~"
                startNo2="~"
            else:  
                tokens=XLtok1.encode(start_token)
                assert len(tokens)==3, "start_token is not a single token in 'tokenizer'"
                startNo=tokens[1]
                tokens=XLtok2.encode(start_token)
                assert len(tokens)==3, "start_token is not a single token in 'tokenizer_2'"
                startNo2=tokens[1]
                embs[tokenNo]=embs[startNo].clone()
                embs2[tokenNo2]=embs2[startNo2].clone()

            # Make a copy of all embeddings to keep all but the embedding[tokenNo] constant 
            index_no_updates = torch.arange(len(embs)) != tokenNo
            orig=embs.clone()
            index_no_updates2 = torch.arange(len(embs2)) != tokenNo2
            orig2=embs2.clone()
    
            print("Begin with '{}'=({}/{}) for '{}'=({}/{})".format(start_token,startNo,startNo2,token,tokenNo,tokenNo2))

            # Create all combinations [prompts] X [promt_variations]
            if prompt_variations:
                token=token+" "
            else:
                prompt_variations=[""]            

            txt_prompts=list()
            for p in prompts:
                for c in prompt_variations:
                    txt_prompts.append(p.format(token+c))
            noPrompts=len(txt_prompts)
            
            # convert imgs to latents
            samples=list()
            for img in imgs:
                samples.append(((XLvae.encode(img.to(XLvae.device)).latent_dist.sample(None))*XLvae.config.scaling_factor).to(XLunet.dtype)) # *XLvae.config.scaling_factor=0.13025:  0.18215    
            noSamples=len(samples)


            # Training Parameters
            batch_size=1
            acc_size=4
            total_steps=sum(i for i, _ in learning_rates)
            # record_every_nth step is recorded in the progression list
            record_every_nth=(total_steps//(intermediate_steps+1)+1)*acc_size
            total_steps*=acc_size

            # Prompt Parametrs
            lora_scale = [0.6]  
            time_ids = torch.tensor(list(imgs[0].shape[2:4])+[0,0]+[1024,1024]).to(XLunet.dtype).to(XLunet.device)


        with torch.enable_grad():
            # Switch Models into training mode
            self.force_training_grad(XLunet,True,True)
            self.force_training_grad(XLt1,True,True)
            self.force_training_grad(XLt2,True,True)
            XLt1.text_model.train()
            XLt2.text_model.train()
            XLunet.train()
            XLunet.enable_gradient_checkpointing()
        
            # Optimizer Parameters        
            learning_rates=iter(learning_rates+[(0,0.0)]) #dummy for last update
            sp,lr=next(learning_rates)
            optimizer = torch.optim.AdamW([embs,embs2], lr=lr, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)   # 1e-7
            optimizer.zero_grad()
                    
            # Progrssion List collects intermediate and final embedding
            progression=list()
            emb=embs[tokenNo].clone()
            emb2=embs2[tokenNo2].clone()
            progression.append({"emb":emb,"emb2":emb2})

            # Display [min (mean) max] of embeddings & current learning rate during training
            desc="[{0:2.3f} ({1:2.3f}) +{2:2.3f}] [{3:2.3f} ({4:2.3f}) +{5:2.3f}] lr={6:1.6f}".format(
                            torch.min(emb.to(float)).detach().cpu().numpy(),
                            torch.mean(emb.to(float)).detach().cpu().numpy(),
                            torch.max(emb.to(float)).detach().cpu().numpy(),
                            torch.min(emb2.to(float)).detach().cpu().numpy(),
                            torch.mean(emb2.to(float)).detach().cpu().numpy(),
                            torch.max(emb2.to(float)).detach().cpu().numpy(),
                            lr)

            # Training Loop
            t=auto.trange(total_steps, desc=desc,leave=True)
            for i in t:
                # use random prompt, random time stepNo, random input image sample
                prompt=txt_prompts[random.randrange(noPrompts)]
                stepNo=torch.tensor(random.randrange(XLsch.config.num_train_timesteps)).unsqueeze(0).long().to(XLunet.device)
                sample=samples[random.randrange(noSamples)].to(XLunet.device)

                ### Target
                noise = torch.randn_like(sample).to(XLunet.device)
                target = noise
                noised_sample=XLsch.add_noise(sample,noise,stepNo)

                # Prediction
                (prompt_embeds,negative_prompt_embeds,pooled_prompt_embeds,negative_pooled_prompt_embeds) = base.encode_prompt(
                    prompt=prompt,prompt_2=prompt,
                    negative_prompt=negative_prompt,negative_prompt_2=negative_prompt,
                    do_classifier_free_guidance=True,lora_scale=lora_scale)
                cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": time_ids}
                pred = XLunet.forward(noised_sample,stepNo,prompt_embeds,added_cond_kwargs=cond_kwargs)['sample']
                            
                # Loss
                loss = torch.nn.functional.mse_loss((pred).float(), (target).float(), reduction="mean")                  
                loss/=float(acc_size)
                loss.backward()

                # One Optimization Step for acc_size gradient accumulation steps
                if ((i+1)%acc_size)==0:
                    # keep Embeddings in normal value range
                    torch.nn.utils.clip_grad_norm_(XLt1.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(XLt2.parameters(), 1.0)

                    optimizer.step()
                    optimizer.zero_grad()

                    with torch.no_grad():                    
                        # keep Embeddings for all other tokens stable      
                        embs[index_no_updates]= orig[index_no_updates]
                        embs2[index_no_updates2]= orig2[index_no_updates2]      
                            
                        # Current Embedding
                        emb=embs[tokenNo].clone()        
                        emb2=embs2[tokenNo2].clone()        
                                
                        if ((i+1)%(record_every_nth))==0:
                            progression.append({"emb":emb,"emb2":emb2})
                            
                        # adjust learning rate?
                        sp-=1
                        if sp<1:
                            sp,lr=next(learning_rates)
                            for g in optimizer.param_groups:
                                g['lr'] = lr
                                
                        # update display
                        t.set_description("[{0:2.3f} ({1:2.3f}) +{2:2.3f}] [{3:2.3f} ({4:2.3f}) +{5:2.3f}] lr={6:1.6f}".format(
                            torch.min(emb.to(float)).detach().cpu().numpy(),
                            torch.mean(emb.to(float)).detach().cpu().numpy(),
                            torch.max(emb.to(float)).detach().cpu().numpy(),
                            torch.min(emb2.to(float)).detach().cpu().numpy(),
                            torch.mean(emb2.to(float)).detach().cpu().numpy(),
                            torch.max(emb2.to(float)).detach().cpu().numpy(),
                            lr))

            # append final Embedding
            progression.append({"emb":emb,"emb2":emb2})
            
            return progression
        

    def create_embedding(self):
        imgs, PILimgs = self.load_imgs(self.imgs_path, wh = self.imgs_wh, flip = self.imgs_flip)


        torch.manual_seed(46)
        progression = self.XL_textual_inversion(self.base,
                                                imgs = imgs,
                                                prompts = self.prompts,
                                                prompt_variations = self.prompt_variations,
                                                token = self.learn_token,
                                                start_token = self.start_token, 
                                                negative_prompt = self.negative_prompt, 
                                                learning_rates = self.learning_rates, 
                                                intermediate_steps = self.intermediate_steps) 
 
        # save final embedding
        self.save_XLembedding(progression[-1], embedding_file = self.emb_file, path=self.embs_path)
        # save intermediate embeddings
        self.save_XLembedding(progression, embedding_file = "all"+self.emb_file, path=self.embs_path)

        # VAE was used in fp32 for training - switch back to fp16
        self.base.vae.to(self.base.unet.dtype)

        progression=torch.load(self.embs_path+"all"+self.emb_file)

        prompt=self.test_prompt.format(self.learn_token)
        seed=1

        frames=list()
        for emb in progression:
            self.set_XLembedding(self.base, emb, token = self.learn_token)
            with torch.no_grad():    
                torch.manual_seed(seed)
                image = self.base(
                    prompt=prompt,
                    negative_prompt=self.negative_prompt,
                    num_inference_steps=40,
                    guidance_scale=7.5
                ).images
            frames.append(image[0])
        
        imageio.mimsave(self.outGIF, frames+[frames[-1]]*2, format='GIF', duration=1.0)



if __name__ == "__main__":
    _ = Teacher("consistent_character/config/erin.yaml")
    _.create_embedding()