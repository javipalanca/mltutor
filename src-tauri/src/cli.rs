//! Command-line flags of the `mltutor` binary.

use std::path::PathBuf;

/// What the binary was asked to do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mode {
    /// Normal interactive session.
    Normal,
    /// Open the window, visit every route and exit (0 ok, 1 failure, 2 timeout).
    SmokeTest,
    /// Save a PNG of the initial window and exit.
    Screenshot(PathBuf),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Run(Mode),
    PrintHelp,
    PrintVersion,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CliError(pub String);

impl std::fmt::Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

pub const USAGE: &str = "\
Uso: mltutor [OPCIONES]

Opciones:
  --smoke-test          Abre la ventana, recorre todas las pantallas y sale
                        (0 = correcto, 1 = fallo, 2 = tiempo agotado)
  --screenshot <png>    Guarda una captura de la ventana inicial y sale
  -h, --help            Muestra esta ayuda
  -V, --version         Muestra la versión";

/// Parses the arguments that follow the program name.
///
/// Unknown arguments are ignored (and logged by the caller) because some launchers add
/// their own, e.g. `-psn_0_12345` when macOS starts an app from Finder.
pub fn parse<I, S>(args: I) -> Result<(Command, Vec<String>), CliError>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut args = args.into_iter().map(Into::into);
    let mut smoke = false;
    let mut screenshot: Option<PathBuf> = None;
    let mut ignored = Vec::new();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => return Ok((Command::PrintHelp, ignored)),
            "-V" | "--version" => return Ok((Command::PrintVersion, ignored)),
            "--smoke-test" => smoke = true,
            "--screenshot" => match args.next() {
                Some(path) if !path.starts_with("--") => screenshot = Some(PathBuf::from(path)),
                _ => {
                    return Err(CliError(
                        "--screenshot necesita la ruta del PNG de salida".into(),
                    ))
                }
            },
            other => match other.strip_prefix("--screenshot=") {
                Some("") => {
                    return Err(CliError(
                        "--screenshot necesita la ruta del PNG de salida".into(),
                    ))
                }
                Some(path) => screenshot = Some(PathBuf::from(path)),
                None => ignored.push(other.to_owned()),
            },
        }
    }

    let mode = match (smoke, screenshot) {
        (true, Some(_)) => {
            return Err(CliError(
                "--smoke-test y --screenshot no se pueden usar a la vez".into(),
            ))
        }
        (true, None) => Mode::SmokeTest,
        (false, Some(path)) => Mode::Screenshot(path),
        (false, None) => Mode::Normal,
    };
    Ok((Command::Run(mode), ignored))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(args: &[&str]) -> Result<Command, CliError> {
        parse(args.iter().copied()).map(|(command, _)| command)
    }

    #[test]
    fn defaults_to_normal_mode() {
        assert_eq!(run(&[]).unwrap(), Command::Run(Mode::Normal));
    }

    #[test]
    fn parses_smoke_test_and_screenshot() {
        assert_eq!(
            run(&["--smoke-test"]).unwrap(),
            Command::Run(Mode::SmokeTest)
        );
        assert_eq!(
            run(&["--screenshot", "out.png"]).unwrap(),
            Command::Run(Mode::Screenshot("out.png".into()))
        );
        assert_eq!(
            run(&["--screenshot=a b.png"]).unwrap(),
            Command::Run(Mode::Screenshot("a b.png".into()))
        );
    }

    #[test]
    fn rejects_invalid_combinations() {
        assert!(run(&["--screenshot"]).is_err());
        assert!(run(&["--screenshot", "--smoke-test"]).is_err());
        assert!(run(&["--screenshot="]).is_err());
        assert!(run(&["--smoke-test", "--screenshot", "x.png"]).is_err());
    }

    #[test]
    fn ignores_unknown_arguments() {
        let (command, ignored) = parse(["-psn_0_1234", "--smoke-test"]).unwrap();
        assert_eq!(command, Command::Run(Mode::SmokeTest));
        assert_eq!(ignored, vec!["-psn_0_1234".to_owned()]);
    }

    #[test]
    fn help_and_version() {
        assert_eq!(run(&["--help"]).unwrap(), Command::PrintHelp);
        assert_eq!(run(&["-V"]).unwrap(), Command::PrintVersion);
    }
}
