use byteorder::{ReadBytesExt, LittleEndian as LE};
use serde::{Deserialize};
use serde_json::value::{Value as JsonValue};
use smol_str::{SmolStr};

use std::collections::{BTreeMap};
use std::fs::{File};
use std::io::{Read, Cursor};
use std::mem::{size_of};
use std::path::{Path};

#[derive(Clone, Copy, Deserialize, Debug)]
#[non_exhaustive]
pub enum SafetensorDtype {
  F64,
  F32,
  I64,
  I32,
  I16,
  I8,
  U64,
  U32,
  U16,
  U8,
  #[serde(rename = "BOOL")]
  Bool,
  F16,
  #[serde(rename = "BF16")]
  Bf16,
}

#[derive(Clone, Deserialize, Debug)]
pub struct SafetensorEntry {
  pub shape: Box<[i64]>,
  pub dtype: SafetensorDtype,
  pub data_offsets: [u64; 2],
}

type SafetensorEntries = BTreeMap<SmolStr, SafetensorEntry>;

#[derive(Clone, Debug)]
pub struct SafetensorHeader {
  pub buf_start: u64,
  pub entries: SafetensorEntries,
  pub raw_metadata: Option<JsonValue>,
}

fn json_value_from_slice_with_trailing<'a>(buf: &'a [u8]) -> Result<JsonValue, serde_json::Error> {
  let mut cur = Cursor::new(buf);
  {
    let mut de = serde_json::de::Deserializer::new(serde_json::de::IoRead::new(&mut cur));
    Deserialize::deserialize(&mut de)
  }
}

impl SafetensorHeader {
  pub fn open<P: AsRef<Path>>(path: P) -> Result<SafetensorHeader, ()> {
    let file = File::open(path.as_ref()).map_err(|_| ())?;
    SafetensorHeader::from_reader(file)
  }

  pub fn from_bytes<'a>(buf: &'a [u8]) -> Result<SafetensorHeader, ()> {
    SafetensorHeader::from_reader(Cursor::new(buf))
  }

  pub fn from_reader<R: Read>(mut reader: R) -> Result<SafetensorHeader, ()> {
    let magic = reader.read_u64::<LE>().map_err(|_| ())?;
    let buf_start = (size_of::<u64>() as u64) + magic;
    let h_sz: usize = magic.try_into().unwrap();
    let mut hbuf = Vec::with_capacity(h_sz);
    hbuf.resize(h_sz, 0);
    reader.read_exact(&mut hbuf).map_err(|_| ())?;
    let raw_value = json_value_from_slice_with_trailing(&hbuf).map_err(|_| ())?;
    let (raw_value, raw_metadata) = match raw_value {
      JsonValue::Object(mut inner) => {
        let raw_metadata = inner.remove_entry("__metadata__").map(|(_, v)| v);
        (JsonValue::Object(inner), raw_metadata)
      }
      _ => (raw_value, None)
    };
    let entries: SafetensorEntries = serde_json::from_value(raw_value).map_err(|_| ())?;
    Ok(SafetensorHeader{
      buf_start: buf_start,
      entries,
      raw_metadata,
    })
  }
}
