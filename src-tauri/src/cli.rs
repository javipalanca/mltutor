//! Command-line flags of the `mltutor` binary.

use std::path::PathBuf;

/// What the binary was asked to do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Mode {
    /// Normal interactive session.
    Normal,
    /// Open the window, visit every route and exit (0 ok, 1 failure, 2 timeout).
    SmokeTest,
    /// Save a PNG of a screen and exit.
    Screenshot(Screenshot),
}

/// `--screenshot <png> [--scenario <name>] [--full-page]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Screenshot {
    pub path: PathBuf,
    /// Screen to prepare before the capture (`ui/src/lib/scenarios.ts`); the initial page
    /// if `None`. Some take an argument after a colon, e.g. `csv_clasificacion:tips.csv`.
    pub scenario: Option<String>,
    /// Grow the window to the height of the page so the PNG holds the whole page.
    pub full_page: bool,
}

#[cfg(test)]
impl Screenshot {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            scenario: None,
            full_page: false,
        }
    }
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
  --screenshot <png>    Guarda una captura de la ventana (PNG) y sale
  --scenario <nombre>   Con --screenshot: pantalla que se prepara antes de capturar
                        (p. ej. datos_iris, datos_pairplot_kde_iris, csv_vacio;
                        csv_clasificacion:<ruta.csv>). Por defecto, la inicial
  --full-page           Con --screenshot: captura la página entera, no solo la ventana
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
    let mut scenario: Option<String> = None;
    let mut full_page = false;
    let mut ignored = Vec::new();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => return Ok((Command::PrintHelp, ignored)),
            "-V" | "--version" => return Ok((Command::PrintVersion, ignored)),
            "--smoke-test" => smoke = true,
            "--full-page" => full_page = true,
            "--scenario" => match args.next() {
                Some(name) if !name.starts_with("--") && !name.is_empty() => {
                    scenario = Some(name);
                }
                _ => {
                    return Err(CliError(
                        "--scenario necesita el nombre de la pantalla".into(),
                    ))
                }
            },
            "--screenshot" => match args.next() {
                Some(path) if !path.starts_with("--") => screenshot = Some(PathBuf::from(path)),
                _ => {
                    return Err(CliError(
                        "--screenshot necesita la ruta del PNG de salida".into(),
                    ))
                }
            },
            other if other.starts_with("--scenario=") => match other.strip_prefix("--scenario=") {
                Some(name) if !name.is_empty() => scenario = Some(name.to_owned()),
                _ => {
                    return Err(CliError(
                        "--scenario necesita el nombre de la pantalla".into(),
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

    if screenshot.is_none() && (scenario.is_some() || full_page) {
        return Err(CliError(
            "--scenario y --full-page solo se usan con --screenshot".into(),
        ));
    }
    let mode = match (smoke, screenshot) {
        (true, Some(_)) => {
            return Err(CliError(
                "--smoke-test y --screenshot no se pueden usar a la vez".into(),
            ))
        }
        (true, None) => Mode::SmokeTest,
        (false, Some(path)) => Mode::Screenshot(Screenshot {
            path,
            scenario,
            full_page,
        }),
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
            Command::Run(Mode::Screenshot(Screenshot::new("out.png")))
        );
        assert_eq!(
            run(&["--screenshot=a b.png"]).unwrap(),
            Command::Run(Mode::Screenshot(Screenshot::new("a b.png")))
        );
        assert_eq!(
            run(&[
                "--screenshot",
                "x.png",
                "--scenario",
                "datos_iris",
                "--full-page"
            ])
            .unwrap(),
            Command::Run(Mode::Screenshot(Screenshot {
                path: "x.png".into(),
                scenario: Some("datos_iris".into()),
                full_page: true,
            }))
        );
        assert_eq!(
            run(&["--scenario=csv_vacio", "--screenshot", "x.png"]).unwrap(),
            Command::Run(Mode::Screenshot(Screenshot {
                path: "x.png".into(),
                scenario: Some("csv_vacio".into()),
                full_page: false,
            }))
        );
    }

    #[test]
    fn rejects_invalid_combinations() {
        assert!(run(&["--screenshot"]).is_err());
        assert!(run(&["--screenshot", "--smoke-test"]).is_err());
        assert!(run(&["--screenshot="]).is_err());
        assert!(run(&["--smoke-test", "--screenshot", "x.png"]).is_err());
        assert!(run(&["--scenario", "datos_iris"]).is_err());
        assert!(run(&["--full-page"]).is_err());
        assert!(run(&["--screenshot", "x.png", "--scenario"]).is_err());
        assert!(run(&["--screenshot", "x.png", "--scenario="]).is_err());
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
