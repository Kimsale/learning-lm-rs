mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;
mod chat;

use std::path::{PathBuf, Path};
use eframe::egui;
use tokenizers::Tokenizer;

#[derive(Default)]
struct App {
    // 界面状态
    model_type: String,
    input_text: String,
    output_text: String,
    is_processing: bool,
    
    // 模型实例
    llama_chat: Option<model::Llama<f32>>,
    llama_story: Option<model::Llama<f32>>,

    // 共享资源
    tokenizer: Option<Tokenizer>,
    cache: Option<kvcache::KVCache<f32>>,

    start_time: Option<std::time::Instant>,  
    elapsed_secs: f32,                       
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::left("control_panel").show(ctx, |ui| {
            ui.heading("Model Control");
            ui.separator();
            
            // 模式选择
            egui::ComboBox::from_label("Operation Mode")
                .selected_text(&self.model_type)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.model_type, "chat".into(), "Conversational Mode");
                    ui.selectable_value(&mut self.model_type, "story".into(), "Continuation mode");
                });
            
            ui.separator();
            
            // 模型加载按钮
            if ui.button("Loading the model").clicked() {
                self.load_model();
            }
            
            ui.separator();
            ui.label("State:");
            if self.llama_chat.is_some() || self.llama_story.is_some() {
                ui.label(" Model loaded");
            } else {
                ui.label(" Model not loaded");
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Text Generation");
            ui.separator();
            
            // 输入区域
            ui.label("Input text:");
            ui.text_edit_multiline(&mut self.input_text);
            
            // 生成按钮
            ui.horizontal(|ui| {
                if ui.button("Generating text").clicked() && !self.is_processing {
                    self.start_generation();
                }
                if self.is_processing {
                    ui.spinner();
                }
            });

            // 生成完成后调用（例如在生成线程结束时）
            self.finalize_generation();
            
            ui.separator();
            
            // 输出区域
            ui.label("Generate results:");
            ui.text_edit_multiline(&mut self.output_text);

            // 在输出区域下方添加
            ui.separator();
            ui.label(format!("🕒 This generation took: {:.1} seconds", self.elapsed_secs));
        });
    }
}

impl App {
    fn load_model(&mut self) {
        let model_path = build_model_path(&self.model_type);
        
        // 加载分词器
        self.tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
            .ok();
        
        // 加载模型
        match self.model_type.as_str() {
            "chat" => {
                self.llama_chat = Some(model::Llama::from_safetensors(model_path));
            }
            "story" => {
                self.llama_story = Some(model::Llama::from_safetensors(model_path));
            }
            _ => {}
        }
    }

    fn start_generation(&mut self) {
        if self.tokenizer.is_none() {
            self.output_text = "Please load the model first".to_string();
            return;
        }

        // 记录开始时间
        self.start_time = Some(std::time::Instant::now());
        let input = self.input_text.clone();
        let tokenizer = self.tokenizer.as_ref().unwrap();
        let mut cache = self.cache.take().unwrap_or_else(|| {
            match self.model_type.as_str() {
                "chat" => self.llama_chat.as_ref().unwrap().new_cache(),
                _ => self.llama_story.as_ref().unwrap().new_cache(),
            }
        });

        // 执行生成
        let binding = tokenizer.encode(&*input, true).unwrap();
        let input_ids = binding.get_ids();
        
        let output_ids = match self.model_type.as_str() {
            "chat" => self.llama_chat.as_ref().unwrap().generate_cache(
                input_ids, 250, 0.8, 30, 0.7, &mut cache
            ),
            _ => self.llama_story.as_ref().unwrap().generate(
                input_ids, 500, 0.8, 30, 0.7
            ),
        };
        
        // 保存缓存
        self.cache = Some(cache);
        
        // 解码结果
        self.output_text = tokenizer.decode(&output_ids, true)
            .unwrap()
            .replace("\n<|im_end|>\n", "")
            .trim()
            .to_string();

    }

    // 在生成完成后调用
    fn finalize_generation(&mut self) {
        // 计算耗时
        if let Some(start) = self.start_time {
            self.elapsed_secs = start.elapsed().as_secs_f32();
        }
    }
}

fn build_model_path(model_type: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join(model_type)
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        // 修正窗口大小设置
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0]),
        ..Default::default()
    };
    
    eframe::run_native(
        "UI PLATFORM", 
        options, 
        //Box::new(|_cc| Box::new(App::default()))
        Box::new(|_cc| Ok(Box::new(App::default())))
    )
}


/// 以下是无UI的实现

// mod config;
// mod kvcache;
// mod model;
// mod operators;
// mod params;
// mod tensor;
// mod chat;

// use std::path::{Path, PathBuf};
// use clap::Parser;
// use tokenizers::Tokenizer;
// use crate::model::Llama;
// use crate::kvcache::KVCache;


// /* fn main() {
//     let project_dir = env!("CARGO_MANIFEST_DIR");
//     let model_dir = PathBuf::from(project_dir).join("models").join("story");
//     let llama = model::Llama::<f32>::from_safetensors(&model_dir);
//     let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    
//     let input = "Once upon a time";
//     let binding = tokenizer.encode(input, true).unwrap();
//     let input_ids = binding.get_ids();

//     //print!("\n{}\n", input);
//     let output_ids = llama.generate(
//         input_ids,
//         500,  //max_len
//         0.8,  //top_p
//         30,   //top_k
//         0.7,   //temperature
//     );
//     println!("{}", tokenizer.decode(&output_ids, true).unwrap());
// } */



// #[derive(Parser, Debug)]
// #[command(version, about)]
// struct Args {
//     #[arg(short, long)]
//     mode: String,

//     #[arg(long, default_value_t = 500)]
//     max_len: usize,

//     #[arg(long, default_value_t = 0.8)]
//     top_p: f32,

//     #[arg(long, default_value_t = 30)]
//     top_k: u32,

//     #[arg(long, default_value_t = 0.7)]
//     temperature: f32,
// }

// fn main() {
//     let args = Args::parse();
    
//     match args.mode.as_str() {
//         "chat" => run_chat_mode(),
//         _ => run_story_mode(&args),
//     }
//     //generate_chat()
// }

// fn generate_chat(    
//     input: String,
//     tokenizer: &Tokenizer,  // 改为引用
//     llama: &Llama<f32>,     // 改为引用
//     cache: &mut KVCache<f32> // 改为可变引用
//     ) -> String {
//     // let mut input = String::new();
//     // std::io::stdin().read_line(&mut input).unwrap();
//     // let user_input = input.trim();
    
//     println!("start generate_chat");
//     // 编码输入
//     // let binding = tokenizer.encode(dialog_history.as_str(), true).unwrap();
//     let binding = tokenizer.encode(input, true).unwrap();
//     let input_ids = binding.get_ids();
    
//     // 生成回复
//     println!("go to generate_cache");
//     let result = llama.generate_cache(
//         input_ids,
//         250,  // 每轮最大生成长度
//         0.8,
//         30,
//         0.7,
//         cache
//     );
//     println!("finish generate_cache");
    
//     // 解码并更新历史
//     let response = tokenizer.decode(&result, true).unwrap();
//     const END_MARKER: &str = "\n<|im_end|>\n";
//     let clean_response = response.replace(END_MARKER, "").trim().to_string();
//     // println!("Assistant: {}", clean_response);
//     // println!("Assistant_result: {}", clean_response);
    
//     // 更新对话历史
//     // dialog_history.push_str(&format!("{}{}", clean_response, END_MARKER));
    
//     // // 缓存管理（限制历史长度）
//     // if dialog_history.len() > 4000 {
//     //     dialog_history.drain(0..2000);
//     //     cache = llama.new_cache(); // 历史过长时重置缓存
//     // }

//     let output_text = clean_response;
//     output_text
//     //println!("Assistant_result: {}", output_text);

// }

// // 对话模式实现
// fn run_chat_mode() {
//     let model_dir = build_model_path("chat");
//     //let llama = Llama::from_safetensors(&model_dir);
//     let llama = Llama::from_safetensors(&model_dir);
//     let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();

//     let mut cache = llama.new_cache();

//     // 对话模板常量
//     const SYSTEM_PROMPT: &str = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
//     const USER_PREFIX: &str = "<|im_start|>user\n";
//     const ASSISTANT_PREFIX: &str = "<|im_start|>assistant\n";
//     const END_MARKER: &str = "\n<|im_end|>\n";

//     println!("进入对话模式(输入exit退出):");
//     loop {
//         let mut input = String::new();
//         std::io::stdin().read_line(&mut input).unwrap();
//         let user_input = input.trim();
        
//         if user_input == "exit" {
//             break;
//         }

//         let input_format = format!("{}{}{}{}{}", SYSTEM_PROMPT, USER_PREFIX, user_input, END_MARKER, ASSISTANT_PREFIX);
//         println!("User:{}",user_input);

//         let response = generate_chat(input_format, &tokenizer, &llama, &mut cache);
//         println!("Assistant: {}", response);
//     }
// }

// // 文本续写模式实现
// fn run_story_mode(args: &Args) {
//     let model_dir = build_model_path("story");
//     let llama = model::Llama::<f32>::from_safetensors(&model_dir);
//     let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
//     .unwrap_or_else(|_| panic!("Story模型分词器加载失败"));

//     // let binding = tokenizer.encode(&args.input, true).unwrap();

//     println!("进入续写模式，请输入要续写的文本:");
//     let mut input = String::new();
//     std::io::stdin().read_line(&mut input).unwrap();
//     let input_text = input.trim();

//     //let input_text: &str = &args.input.as_str();
//     let binding = tokenizer.encode(input_text, true).unwrap();

//     let input_ids = binding.get_ids();

//     let output_ids = llama.generate(
//         input_ids,
//         args.max_len,
//         args.top_p,
//         args.top_k,
//         args.temperature,
//     );

//     println!("生成结果:\n{}", tokenizer.decode(&output_ids, true).unwrap());
// }

// // 公共工具函数
// fn build_model_path(mode: &str) -> PathBuf {
//     PathBuf::from(env!("CARGO_MANIFEST_DIR"))
//         .join("models")
//         .join(mode)
// }
