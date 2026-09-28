-- 0002: allow 'changelog' documents.
--
-- WorkHorse gained a root CHANGELOG.md and desk/CHANGELOG.md (2026-09-19), each entry naming the
-- release it follows. Studbook's one answer judged unfaithful in every dev run (wh-15) misordered
-- releases because no passage stated the order (docs/refusal-experiments.md); the changelogs do.
--
-- Widening only: every one of the 17 values in 0001's check is carried over verbatim, in the same
-- order, and 'changelog' is appended. Nothing is removed. The previous definition, read from the
-- live catalog before writing this (pg_get_constraintdef on documents_doc_type_check), was
-- exactly 0001's list.
alter table studbook.documents drop constraint documents_doc_type_check;
alter table studbook.documents add constraint documents_doc_type_check check (doc_type in (
  'intent', 'spec', 'adr', 'evals', 'plan', 'verification', 'review-packet', 'release',
  'approvals', 'retro', 'conductor-log', 'instinct', 'codebase-map', 'constraints',
  'design-doc', 'readme', 'commit',
  'changelog'
));
