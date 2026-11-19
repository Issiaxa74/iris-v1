//! Creación instintiva de memoria (TOJI_SPEC.md §2) — versión de cabrita.
//!
//! En Iris, neuronas, satélites y cables nacen durante el pensamiento profundo
//! antes de responder. Cabrita no tiene ese motor todavía, así que hace la
//! versión mínima y honesta: al procesar el turno activa las neuronas cuyas
//! palabras aparecen en el texto, añade un satélite (evento o emoción) a las
//! más fuertes, refuerza el cable entre las coactivadas y registra la frontera
//! del sector. El motor definitivo puede sustituir `MemoryUpdater` sin cambiar
//! el formato `.mem`.

use crate::toji::{
    Border, Cable, Neuron, Satellite, Sector, CABLE_SIMILITUD, SAT_EMOTION, SAT_EVENT,
};
use crate::util::{now_unix, sector_name, words};

const MAX_WORDS: usize = 16;
const MAX_NEW_NEURONS: usize = 6;
const MAX_ACTIVATED: usize = 8;
const MAX_CABLES: usize = 6;

const EMOTION_CUES: &[&str] = &[
    "gracias",
    "genial",
    "increible",
    "mal",
    "fatal",
    "odio",
    "amo",
    "miedo",
    "error",
    "enfadado",
    "molesto",
    "triste",
    "feliz",
    "orgulloso",
    "preocupado",
    "excelente",
];

/// Neurona activada que se pasa al motor de respuesta.
#[derive(Debug, Clone)]
pub struct NeuronView {
    pub id: u32,
    pub word: String,
    pub satellites: usize,
    pub strength: f32,
    pub activation: f32,
}

/// Estado de memoria que ve el motor en este turno.
#[derive(Debug, Clone, Default)]
pub struct MemoryView {
    pub sector: String,
    pub neurons_total: usize,
    pub cables_total: usize,
    pub activated: Vec<NeuronView>,
    pub created_neurons: usize,
    pub satellites_added: usize,
}

impl MemoryView {
    #[allow(dead_code)] // constructor explícito usado por tests y por el motor sin TOJI
    pub fn empty(sector: impl Into<String>) -> Self {
        MemoryView {
            sector: sector.into(),
            ..Default::default()
        }
    }

    pub fn summary(&self) -> String {
        if self.activated.is_empty() {
            format!(
                "TOJI[{}]: {} neuronas, sin activación",
                self.sector, self.neurons_total
            )
        } else {
            let top: Vec<&str> = self
                .activated
                .iter()
                .take(4)
                .map(|n| n.word.as_str())
                .collect();
            format!(
                "TOJI[{}]: {} neuronas, {} cables, activas: {}",
                self.sector,
                self.neurons_total,
                self.cables_total,
                top.join(", ")
            )
        }
    }
}

pub struct MemoryUpdater {
    pub max_words: usize,
    pub max_new_neurons: usize,
}

impl Default for MemoryUpdater {
    fn default() -> Self {
        MemoryUpdater {
            max_words: MAX_WORDS,
            max_new_neurons: MAX_NEW_NEURONS,
        }
    }
}

impl MemoryUpdater {
    /// Registra el turno en el sector y devuelve lo activado para el motor.
    pub fn absorb(&self, sector: &mut Sector, user_text: &str) -> MemoryView {
        let mut view = MemoryView {
            sector: sector.name.clone(),
            neurons_total: sector.neurons.len(),
            cables_total: sector.cables.len(),
            ..Default::default()
        };

        let mut uniq: Vec<String> = Vec::new();
        for w in words(user_text, 3) {
            if uniq.len() >= self.max_words {
                break;
            }
            if !uniq.contains(&w) {
                uniq.push(w);
            }
        }
        let epoch = now_unix();
        let emotion = EMOTION_CUES
            .iter()
            .any(|c| crate::util::normalize(user_text).contains(c));
        let sat_text = truncate(user_text.trim(), 240);
        let mut seen: Vec<usize> = Vec::new();

        for w in &uniq {
            match sector.find_neuron(w) {
                Some(idx) => {
                    seen.push(idx);
                    let n = &mut sector.neurons[idx];
                    if !n.satellites.iter().any(|s| s.text == sat_text) {
                        n.satellites.push(Satellite {
                            t_epoch: epoch,
                            kind: if emotion { SAT_EMOTION } else { SAT_EVENT },
                            strength: 0.35,
                            text: sat_text.clone(),
                        });
                        view.satellites_added += 1;
                    }
                }
                None if view.created_neurons < self.max_new_neurons => {
                    let id = sector.next_id();
                    let mut n = Neuron::new(id, w.clone());
                    n.satellites.push(Satellite {
                        t_epoch: epoch,
                        kind: if emotion { SAT_EMOTION } else { SAT_EVENT },
                        strength: 0.4,
                        text: sat_text.clone(),
                    });
                    sector.neurons.push(n);
                    seen.push(sector.neurons.len() - 1);
                    view.created_neurons += 1;
                    view.satellites_added += 1;
                }
                None => {}
            }
        }

        // Cables entre coactivadas: nacen al usarse juntas (similitud contextual).
        let mut made = 0usize;
        'outer: for i in 0..seen.len() {
            for j in (i + 1)..seen.len() {
                if made >= MAX_CABLES {
                    break 'outer;
                }
                let a = sector.neurons[seen[i]].id;
                let b = sector.neurons[seen[j]].id;
                if a == b {
                    continue;
                }
                match sector.cables.iter_mut().find(|c| c.same_pair(a, b)) {
                    Some(c) => c.weight = (c.weight + 0.05).min(1.0),
                    None => {
                        sector.cables.push(Cable {
                            a,
                            b,
                            kind: CABLE_SIMILITUD,
                            weight: 0.5,
                        });
                        made += 1;
                    }
                }
            }
        }

        // Frontera con el sector general (quién limita con quién).
        let name = sector_name(&sector.name);
        if name != "general" {
            let exists = sector.borders.iter().any(|b| {
                let (x, y) = (sector_name(&b.sector_a), sector_name(&b.sector_b));
                (x == name && y == "general") || (x == "general" && y == name)
            });
            if !exists {
                sector.borders.push(Border {
                    sector_a: name,
                    sector_b: "general".to_string(),
                    weight: 0.5,
                });
            }
        }

        let mut activated: Vec<NeuronView> = seen
            .iter()
            .map(|idx| {
                let n = &sector.neurons[*idx];
                NeuronView {
                    id: n.id,
                    word: n.id_str.clone(),
                    satellites: n.satellites.len(),
                    strength: n.strength(),
                    activation: 1.0 + 0.2 * n.satellites.len() as f32,
                }
            })
            .collect();
        activated.sort_by(|a, b| {
            b.activation
                .partial_cmp(&a.activation)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        activated.truncate(MAX_ACTIVATED);

        view.activated = activated;
        view.neurons_total = sector.neurons.len();
        view.cables_total = sector.cables.len();
        view
    }
}

fn truncate(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crea_neurona_y_satelite() {
        let mut s = Sector::new("programacion");
        let up = MemoryUpdater::default();
        let v = up.absorb(&mut s, "estoy programando en rust con cabrita");
        assert!(v.created_neurons >= 2);
        assert!(s.find_neuron("rust").is_some());
        assert!(v.satellites_added >= 1);
        assert!(!v.activated.is_empty());
    }

    #[test]
    fn segunda_visita_refuerza_sin_duplicar_satelite() {
        let mut s = Sector::new("programacion");
        let up = MemoryUpdater::default();
        up.absorb(&mut s, "rust ownership");
        let before = s.neurons.len();
        let v = up.absorb(&mut s, "rust ownership");
        assert_eq!(s.neurons.len(), before);
        assert_eq!(v.created_neurons, 0);
        assert_eq!(
            v.satellites_added, 0,
            "el mismo texto no debe duplicar el satélite"
        );
        assert!(s.cables.len() >= 1);
    }

    #[test]
    fn detecta_emocion() {
        let mut s = Sector::new("general");
        let up = MemoryUpdater::default();
        up.absorb(&mut s, "muchas gracias, fue genial");
        let has_emotion = s
            .neurons
            .iter()
            .flat_map(|n| n.satellites.iter())
            .any(|sat| sat.kind == SAT_EMOTION);
        assert!(has_emotion);
    }

    #[test]
    fn frontera_con_general() {
        let mut s = Sector::new("matematicas");
        MemoryUpdater::default().absorb(&mut s, "integral definida");
        assert!(s.borders.iter().any(|b| b.sector_b == "general"));
    }
}
