extern crate byteorder;
extern crate libc;
extern crate serde;
extern crate serde_json;
extern crate smol_str;

use crate::sys::{MmapFile};

use byteorder::{ReadBytesExt};
use serde::{Deserialize};
use serde_json::value::{Value as JsonValue};

use std::io::{Read, Cursor};
use std::path::{Path};

pub mod safetensor;
pub mod sys;

pub const MAGIC: &'static [u8; 8] = b"10SRSAFE";

#[derive(Clone, Copy, Deserialize, Debug)]
#[non_exhaustive]
pub enum TensorsafeFormat {
  #[serde(rename = "nd")]
  Nd,
}

#[derive(Clone, Copy, Deserialize, Debug)]
#[non_exhaustive]
pub enum TensorsafeDtype {
  #[serde(rename = "f32")]
  F32,
  #[serde(rename = "f64")]
  F64,
  #[serde(rename = "i8")]
  I8,
  #[serde(rename = "i16")]
  I16,
  #[serde(rename = "i32")]
  I32,
  #[serde(rename = "i64")]
  I64,
  #[serde(rename = "u8")]
  U8,
  #[serde(rename = "u16")]
  U16,
  #[serde(rename = "u32")]
  U32,
  #[serde(rename = "u64")]
  U64,
  #[serde(rename = "bool")]
  Bool,
  #[serde(rename = "f16")]
  F16,
  #[serde(rename = "bf16")]
  Bf16,
}

#[derive(Clone, Deserialize, Debug)]
pub struct TensorsafeEntry {
  pub start: u64,
  pub end: u64,
  pub fmt: TensorsafeFormat,
  pub shape: Box<[i64]>,
  pub dtype: TensorsafeDtype,
}

#[derive(Clone, Debug)]
pub struct TensorsafeTrailer {
  pub entries: Vec<TensorsafeEntry>,
}

fn jsonl_values_from_slice_with_trailing<'a>(buf: &'a [u8]) -> Result<Vec<JsonValue>, serde_json::Error> {
  let mut vals = Vec::new();
  let mut cur = Cursor::new(buf);
  loop {
    let mut de = serde_json::de::Deserializer::new(serde_json::de::IoRead::new(&mut cur));
    match Deserialize::deserialize(&mut de) {
      Err(_) => {
        return Ok(vals);
      }
      Ok(v) => {
        vals.push(v);
      }
    }
    let mut retry = false;
    loop {
      match cur.read_u8() {
        Err(_) => {
          return Ok(vals);
        }
        Ok(x) => {
          if x == b'\n' {
            break;
          } else {
            if x == b'\r' && !retry {
              retry = true;
            } else {
              return Ok(vals);
            }
          }
        }
      }
    }
  }
}

impl TensorsafeTrailer {
  pub fn from_bytes<'a>(buf: &'a [u8]) -> Result<TensorsafeTrailer, ()> {
    if buf.len() % 16 != 0 {
      return Err(());
    }
    if buf.len() < 16 {
      return Err(());
    }
    if &buf[buf.len() - 8 .. ] != MAGIC {
      return Err(());
    }
    let toff: &[u8; 4] = &buf[buf.len() - 12 .. buf.len() - 8].try_into().unwrap();
    let toff: u32 = u32::from_le_bytes(*toff);
    let raw_vals = jsonl_values_from_slice_with_trailing(&buf[buf.len() - toff as usize .. buf.len() - 16]).map_err(|_| ())?;
    let mut entries = Vec::new();
    for raw_val in raw_vals.into_iter() {
      let e: TensorsafeEntry = serde_json::from_value(raw_val).map_err(|_| ())?;
      entries.push(e);
    }
    Ok(TensorsafeTrailer{entries})
  }
}

#[derive(Clone)]
pub struct Tensorsafe {
  mmap: MmapFile,
  trail: TensorsafeTrailer,
}

impl Tensorsafe {
  pub fn open<P: AsRef<Path>>(path: P) -> Result<Tensorsafe, ()> {
    let mmap = MmapFile::open(path)?;
    let buf = mmap.as_bytes();
    let trail = TensorsafeTrailer::from_bytes(buf)?;
    Ok(Tensorsafe{
      mmap,
      trail,
    })
  }

  pub fn trailer(&self) -> &TensorsafeTrailer {
    &self.trail
  }
}
