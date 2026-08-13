//! Material ids and the property table. The one module outside `plugin/` allowed to
//! use `bevy::platform::collections::HashMap` (load-time name lookup only).

use bevy::platform::collections::HashMap;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Default)]
pub struct MaterialId(pub u8);

pub const AIR: MaterialId = MaterialId(0);

#[derive(Copy, Clone, PartialEq, Eq, Debug, Default, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaterialClass {
    #[default]
    Empty,
    Powder,
    Liquid,
    Gas,
    Solid,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct Material {
    pub name: String,
    pub class: MaterialClass,
    /// TOML `"#c2b280"`; read by pixelview, never by the solver
    #[serde(deserialize_with = "de_hex_color")]
    pub color: [u8; 3],
    /// bulk, kg/m^3
    pub density: f32,
    /// 0.0 for non-powders
    pub grain_density: f32,
    /// water = 1.0
    pub viscosity: f32,
    /// degrees, dry
    pub repose_angle: f32,
    /// µm
    pub grain_size: f32,
    /// aperture µm; 0 = impermeable; f32::INFINITY for air
    pub porosity: f32,
    /// How readily the material soaks up water, 0..1; 0 = never gets wet.
    ///
    /// Deliberately *not* `porosity`, which is an aperture in µm for sieving and
    /// filtration — `passes_powder` asks whether a grain fits through it. The two are
    /// unrelated: steel mesh has a 250 µm aperture and soaks up nothing, cardboard has no
    /// aperture at all and soaks up plenty. Reusing `porosity` as the absorbency flag is
    /// what kept cardboard waterproof, and flipping that test to include solids would have
    /// turned mesh into a sponge.
    ///
    /// Defaulted, so entries authored before it existed — and the tests that build their
    /// own tables — stay valid. Powders still opt in through `porosity`, so their behaviour
    /// is untouched; this is how a *solid* opts in.
    #[serde(default)]
    pub absorbency: f32,
    /// Mohs-ish 1..10; >= 10.0 == undiggable
    pub hardness: f32,
    pub melt_pt: f32,
    pub boil_pt: f32,
    pub freeze_pt: f32,
    pub ignition_pt: f32,
    pub burn_products: Vec<String>,
    pub thermal_cond: f32,
    pub heat_capacity: f32,
    pub elec_cond: f32,
    pub solubility: f32,
    pub reactivity_tags: Vec<String>,
}

/// `"#c2b280"` -> `[0xc2, 0xb2, 0x80]`
fn de_hex_color<'de, D: serde::Deserializer<'de>>(d: D) -> Result<[u8; 3], D::Error> {
    use serde::Deserialize as _;
    let s = String::deserialize(d)?;
    let hex = s.strip_prefix('#').unwrap_or(&s);
    if hex.len() != 6 {
        return Err(serde::de::Error::custom("expected #rrggbb"));
    }
    let mut out = [0u8; 3];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16)
            .map_err(|_| serde::de::Error::custom("expected #rrggbb"))?;
    }
    Ok(out)
}

#[derive(serde::Deserialize)]
struct MaterialFile {
    material: Vec<Material>,
}

pub struct MaterialTable {
    entries: Vec<Material>,
    /// load-time only
    by_name: HashMap<String, MaterialId>,
    // hot SoA mirrors, indexed by material id — the solver reads ONLY these:
    class: [MaterialClass; 256],
    density: [f32; 256],
    viscosity: [f32; 256],
    repose: [f32; 256],
    grain: [f32; 256],
    porosity: [f32; 256],
    absorbency: [f32; 256],
    hardness: [f32; 256],
    dispersion: [u8; 256],
    p_move: [f32; 256],
    /// grain density where authored, else bulk — the Stokes settle input
    settle_density: [f32; 256],
}

impl MaterialTable {
    pub fn from_toml_str(src: &str) -> Result<Self, MaterialError> {
        let file: MaterialFile =
            toml::from_str(src).map_err(|e| MaterialError::Parse(e.to_string()))?;
        Self::from_entries(file.material)
    }

    /// Builds and validates the hot mirrors from an already-parsed entry list.
    pub fn from_entries(entries: Vec<Material>) -> Result<Self, MaterialError> {
        if entries.len() > 256 {
            return Err(MaterialError::TooMany);
        }
        match entries.first() {
            Some(m) if m.name == "air" && m.class == MaterialClass::Empty => {}
            _ => return Err(MaterialError::AirNotFirst),
        }

        let mut by_name: HashMap<String, MaterialId> = HashMap::default();
        let mut table = Self {
            entries: Vec::new(),
            by_name: HashMap::default(),
            class: [MaterialClass::Empty; 256],
            density: [0.0; 256],
            viscosity: [0.0; 256],
            repose: [0.0; 256],
            grain: [0.0; 256],
            // an unauthored id is treated as impermeable, not as a hole
            porosity: [0.0; 256],
            absorbency: [0.0; 256],
            hardness: [0.0; 256],
            dispersion: [0; 256],
            p_move: [0.0; 256],
            settle_density: [0.0; 256],
        };

        for (i, m) in entries.iter().enumerate() {
            if by_name
                .insert(m.name.clone(), MaterialId(i as u8))
                .is_some()
            {
                return Err(MaterialError::DuplicateName(m.name.clone()));
            }
            if i != 0 && !(m.density > 0.0) {
                return Err(MaterialError::BadValue("density must be > 0"));
            }
            if m.porosity < 0.0 || m.porosity.is_nan() {
                return Err(MaterialError::BadValue("porosity must be >= 0"));
            }
            if !(0.0..=1.0).contains(&m.absorbency) {
                return Err(MaterialError::BadValue("absorbency must be 0..1"));
            }
            if m.grain_size < 0.0 {
                return Err(MaterialError::BadValue("grain_size must be >= 0"));
            }
            if m.class == MaterialClass::Powder && !(5.0..=85.0).contains(&m.repose_angle) {
                return Err(MaterialError::BadValue("powder repose_angle must be 5..85"));
            }
            if m.class == MaterialClass::Liquid && !(m.viscosity > 0.0) {
                return Err(MaterialError::BadValue("liquid viscosity must be > 0"));
            }

            table.class[i] = m.class;
            table.density[i] = m.density;
            table.viscosity[i] = m.viscosity;
            table.repose[i] = m.repose_angle;
            table.grain[i] = m.grain_size;
            table.porosity[i] = m.porosity;
            table.absorbency[i] = m.absorbency;
            table.hardness[i] = m.hardness;
            table.settle_density[i] = if m.grain_density > 0.0 {
                m.grain_density
            } else {
                m.density
            };
            // derived liquid mobility: thin liquids race, thick ones creep
            let mu = m.viscosity;
            table.dispersion[i] = if mu > 0.0 {
                (5.0 / mu).round().clamp(0.0, 8.0) as u8
            } else {
                0
            };
            table.p_move[i] = if mu > 0.0 { (5.0 / mu).min(1.0) } else { 0.0 };
        }

        table.entries = entries;
        table.by_name = by_name;
        Ok(table)
    }

    /// `include_str!("../data/materials.toml")`, panics on bad build-time data
    pub fn embedded() -> Self {
        Self::from_toml_str(include_str!("../data/materials.toml"))
            .expect("embedded data/materials.toml is malformed")
    }

    pub fn id(&self, name: &str) -> Option<MaterialId> {
        self.by_name.get(name).copied()
    }

    pub fn get(&self, id: MaterialId) -> &Material {
        &self.entries[id.0 as usize]
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_authored(&self, m: u8) -> bool {
        (m as usize) < self.entries.len()
    }

    /// All authored ids, for callers that need to enumerate (palette bake, spawners).
    pub fn iter(&self) -> impl Iterator<Item = (MaterialId, &Material)> + '_ {
        self.entries
            .iter()
            .enumerate()
            .map(|(i, m)| (MaterialId(i as u8), m))
    }

    #[inline]
    pub fn class(&self, m: u8) -> MaterialClass {
        self.class[m as usize]
    }

    #[inline]
    pub fn density(&self, m: u8) -> f32 {
        self.density[m as usize]
    }

    #[inline]
    pub fn viscosity(&self, m: u8) -> f32 {
        self.viscosity[m as usize]
    }

    #[inline]
    pub fn repose(&self, m: u8) -> f32 {
        self.repose[m as usize]
    }

    #[inline]
    pub fn grain(&self, m: u8) -> f32 {
        self.grain[m as usize]
    }

    #[inline]
    pub fn porosity(&self, m: u8) -> f32 {
        self.porosity[m as usize]
    }

    #[inline]
    pub fn absorbency(&self, m: u8) -> f32 {
        self.absorbency[m as usize]
    }

    #[inline]
    pub fn hardness(&self, m: u8) -> f32 {
        self.hardness[m as usize]
    }

    #[inline]
    pub fn dispersion(&self, m: u8) -> u8 {
        self.dispersion[m as usize]
    }

    #[inline]
    pub fn p_move(&self, m: u8) -> f32 {
        self.p_move[m as usize]
    }

    /// grain density where authored, else bulk — used by the Stokes settle velocity
    #[inline]
    pub fn density_grain_or_bulk(&self, m: u8) -> f32 {
        self.settle_density[m as usize]
    }

    #[inline]
    pub fn is_empty(&self, m: u8) -> bool {
        m == 0
    }

    #[inline]
    pub fn passes_liquid(&self, m: u8) -> bool {
        self.porosity(m) > 0.0
    }

    #[inline]
    pub fn passes_powder(&self, solid: u8, powder: u8) -> bool {
        self.grain(powder) <= self.porosity(solid)
    }

    /// wet-derived bulk density, `w = wetness / 255`
    #[inline]
    pub fn wet_density(&self, m: u8, w: f32) -> f32 {
        self.density(m) + 290.0 * w
    }

    /// wet-derived angle of repose, `w = wetness / 255`
    #[inline]
    pub fn wet_repose(&self, m: u8, w: f32) -> f32 {
        self.repose(m) + 14.0 * w
    }

    /// `true` when `m` occupies its cell — powders and liquids cannot enter it, and
    /// it never joins the active set.
    #[inline]
    pub fn is_solid(&self, m: u8) -> bool {
        self.class(m) == MaterialClass::Solid
    }

    /// The liquid a drip re-emits: the lowest-id `Liquid` entry, `0` when the table
    /// has none. Data-driven, so `wet.rs` never has to name a material.
    pub fn wetting_liquid(&self) -> u8 {
        self.entries
            .iter()
            .position(|m| m.class == MaterialClass::Liquid)
            .unwrap_or(0) as u8
    }

    /// `true` when a powder/liquid cell should be visited by a solver pass
    #[inline]
    pub fn is_mobile(&self, m: u8) -> bool {
        matches!(self.class(m), MaterialClass::Powder | MaterialClass::Liquid)
    }
}

#[derive(Debug, Clone)]
pub enum MaterialError {
    Parse(String),
    TooMany,
    DuplicateName(String),
    AirNotFirst,
    BadColor(String),
    BadValue(&'static str),
}

impl core::fmt::Display for MaterialError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Parse(s) => write!(f, "material table parse error: {s}"),
            Self::TooMany => write!(f, "material table has more than 256 entries"),
            Self::DuplicateName(n) => write!(f, "duplicate material name `{n}`"),
            Self::AirNotFirst => write!(f, "entry 0 must be `air` with class = \"empty\""),
            Self::BadColor(s) => write!(f, "bad colour `{s}`, expected #rrggbb"),
            Self::BadValue(s) => write!(f, "bad material value: {s}"),
        }
    }
}

impl std::error::Error for MaterialError {}
