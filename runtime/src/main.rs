//! CABRITA — runtime de Iris.
//!
//! Carga el modelo UKJ con mmap parcial, tokeniza con IRISTOK, elige ruta,
//! actualiza la memoria TOJI del sector y deja los pares de feedback en la cola
//! que consume el entrenador. No entrena nada y no reescribe el UKJ.

mod cli;
mod engine;
mod feedback;
mod router;
mod toji;
mod tok;
mod ui;
mod ukj;
mod util;

use std::process::ExitCode;

fn main() -> ExitCode {
    let cfg = match cli::parse(std::env::args().skip(1)) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    if cfg.help {
        print!("{}", cli::usage());
        return ExitCode::SUCCESS;
    }
    if cfg.version {
        println!("cabrita {}", cli::VERSION);
        return ExitCode::SUCCESS;
    }

    let ask = cfg.ask.clone();
    let write_memory = cfg.write_memory;
    let mut app = match ui::App::new(cfg) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e:#}");
            return ExitCode::from(2);
        }
    };

    let result = match ask {
        Some(text) => app.ask_once(&text),
        None => app.run_interactive(),
    };
    if let Err(e) = result {
        eprintln!("error: {e:#}");
        return ExitCode::FAILURE;
    }
    if !write_memory {
        // Modo lectura: nada que avisar, pero deja constancia explícita.
        eprintln!("(modo lectura: no se escribieron .mem ni sidecar de estrés)");
    }
    ExitCode::SUCCESS
}
