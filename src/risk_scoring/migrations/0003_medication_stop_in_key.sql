-- Widen the medication natural key to include stop.
--
-- Synthea emits a single dispense and a continuing course of the same drug,
-- ordered at the same encounter and instant, as two rows that differ only in
-- stop and the cost columns the ingestion payload drops. The frozen
-- populations contain 1,001 such pairs in baseline, 1,192 in care_protocol,
-- and 2,136 in demographic_shift; under the narrower key the second row of
-- each pair was rejected as a divergent re-post, and absorbing it instead
-- would have undercounted active medications on 33 baseline discharges.
--
-- The five-column key is unique across all three frozen populations. It is
-- the whole medication payload, so a medication event can no longer conflict:
-- any difference makes it a distinct row. Conditions keep the four-column key
-- and its conflict detection, because they carry system and description
-- outside the key and show no collisions in any population.

ALTER TABLE medications DROP CONSTRAINT medications_pkey;

ALTER TABLE medications ADD PRIMARY KEY (patient, encounter, code, start, stop);
