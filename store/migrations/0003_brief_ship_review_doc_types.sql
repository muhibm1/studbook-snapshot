-- 0003: allow 'brief', 'ship' and 'review' documents.
--
-- WorkHorse 0.3.0 (2026-09-19) replaced intent.md and review-packet.md with brief.md (the Design
-- gate document, carrying the change's "Risk tier: N" line) and ship.md (the Ship gate document),
-- and each reviewer now writes its own reviews/<agent>.md
-- (docs/superpowers/specs/2026-09-19-two-gates-design.md, section 4). 'intent' and
-- 'review-packet' stay: changes made before 0.3.0 keep those files.
--
-- Widening only: every one of the 18 values in 0002's check is carried over verbatim, in the same
-- order, and 'brief', 'ship', 'review' are appended. Nothing is removed. The previous definition,
-- read from the live catalog of both projects before writing this (pg_get_constraintdef on
-- documents_doc_type_check, production productionprojectref and test testprojectrefabcdef, each
-- with 0001 and 0002 in studbook_meta.migrations), was exactly 0002's list.
alter table studbook.documents drop constraint documents_doc_type_check;
alter table studbook.documents add constraint documents_doc_type_check check (doc_type in (
  'intent', 'spec', 'adr', 'evals', 'plan', 'verification', 'review-packet', 'release',
  'approvals', 'retro', 'conductor-log', 'instinct', 'codebase-map', 'constraints',
  'design-doc', 'readme', 'commit',
  'changelog',
  'brief', 'ship', 'review'
));
