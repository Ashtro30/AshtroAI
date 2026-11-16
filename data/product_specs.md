# Aurora Sensor Hub - Product Specification

The Aurora Sensor Hub is Meridian Labs' battery-powered, long-range environmental monitoring node for industrial IoT deployments. Current production firmware: **v2.4.1**.

## Hardware

- Battery: 4,800 mAh lithium thionyl-chloride cell (non-rechargeable, field-replaceable).
- Radio: LoRa, 868/915 MHz, range up to **12 km** line-of-sight (about 2 km in dense urban areas).
- Operating temperature: -20 °C to 60 °C. Ingress protection: IP67.
- Onboard sensors: temperature, humidity, barometric pressure, 3-axis vibration, ambient light, and a magnetic reed switch.
- Processor: ARM Cortex-M4 at 64 MHz with 512 KB flash and 128 KB RAM.

## Power Profile

- Active (sensing + transmitting) current: **12 mA**.
- Deep-sleep current: **0.05 mA**.
- Default duty cycle: **2%** of the time in active mode, 98% asleep.
- Average current is the duty-cycle-weighted mean of the active and sleep currents.

## Security

- AES-128 encryption on all LoRa payloads, with per-device keys provisioned at manufacture.
- Secure boot with signed firmware images; over-the-air updates are verified before flashing.
- Device keys must be rotated at least every 90 days (see the Information Security Policy).

## Pricing and Volume Discounts

List price: **USD 249 per unit**.

| Order size (units) | Discount |
| ------------------ | -------- |
| 1-49               | 0%       |
| 50-199             | 8%       |
| 200 and above      | 15%      |

Discounts apply to the whole order, not just the units above the threshold.

## Warranty and Support

24-month limited hardware warranty. Firmware support is guaranteed for 5 years from the first shipment date. Support requests are triaged using the severity levels in the Incident Response Runbook.
