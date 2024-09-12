use argmin::core::ArgminFloat;
use itertools::Itertools;
use nalgebra::Const;
use nalgebra::U3;

use crate::misc::EndPoints;
use crate::prelude::Contains;
use crate::region::CompoundCurve;
use crate::region::Region;
use crate::{curve::NurbsCurve, misc::FloatingPoint, prelude::CurveIntersectionSolverOptions};

use super::clip::Clip;
use super::operation::BooleanOperation;
use super::Boolean;

/// Boolean operation for Region & NURBS curve
impl<'a, T: FloatingPoint + ArgminFloat> Boolean<&'a NurbsCurve<T, U3>> for Region<T> {
    type Output = anyhow::Result<Clip<T>>;
    type Option = Option<CurveIntersectionSolverOptions<T>>;

    fn boolean(
        &self,
        operation: BooleanOperation,
        other: &'a NurbsCurve<T, Const<3>>,
        option: Self::Option,
    ) -> Self::Output {
        let compound: CompoundCurve<T, U3> = other.clone().into();
        self.boolean(operation, &compound, option)
    }
}

/// Boolean operation for Region & NURBS curve
impl<'a, T: FloatingPoint + ArgminFloat> Boolean<&'a CompoundCurve<T, U3>> for Region<T> {
    type Output = anyhow::Result<Clip<T>>;
    type Option = Option<CurveIntersectionSolverOptions<T>>;

    fn boolean(
        &self,
        operation: BooleanOperation,
        other: &'a CompoundCurve<T, Const<3>>,
        option: Self::Option,
    ) -> Self::Output {
        match operation {
            BooleanOperation::Union => {
                let exterior =
                    self.exterior()
                        .boolean(BooleanOperation::Union, other, option.clone())?;

                let exterior = exterior
                    .into_regions()
                    .into_iter()
                    .next()
                    .map(|region| region.into_exterior())
                    .ok_or(anyhow::anyhow!("No exterior region found"))?;

                let interior_differences = self
                    .interiors()
                    .iter()
                    .map(|interior| {
                        interior.boolean(BooleanOperation::Difference, other, option.clone())
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?;

                let interiors = interior_differences
                    .into_iter()
                    .flat_map(|c| {
                        c.into_regions()
                            .into_iter()
                            .map(|region| region.into_exterior())
                    })
                    .collect();

                Ok(Clip::new(
                    vec![Region::new(exterior, interiors)],
                    Default::default(),
                ))
            }
            BooleanOperation::Intersection => {
                // difference other and interiors
                let mut clip_regions: Vec<Region<T>> = vec![other.clone().into()];
                for it in self.interiors().iter() {
                    let mut next_clips = vec![];
                    for clip in clip_regions.iter() {
                        let c = clip.boolean(BooleanOperation::Difference, it, option.clone())?;
                        let mut regions = c.into_regions();
                        let mut interiors = clip.interiors().iter().cloned().collect_vec();
                        distribute_interiors_to_regions(
                            &mut regions,
                            &mut interiors,
                            option.clone(),
                        )?;
                        next_clips.extend(regions);
                    }
                    clip_regions = next_clips;
                }

                let regions = clip_regions
                    .into_iter()
                    .map(|clip_region| {
                        let (ex, mut its) = clip_region.into_tuple();
                        self.exterior()
                            .boolean(BooleanOperation::Intersection, &ex, option.clone())
                            .and_then(|clip| {
                                let mut regions = clip.into_regions();
                                distribute_interiors_to_regions(
                                    &mut regions,
                                    &mut its,
                                    option.clone(),
                                )?;
                                Ok(regions)
                            })
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?
                    .into_iter()
                    .flatten()
                    .collect();

                Ok(Clip::new(regions, Default::default()))
            }
            BooleanOperation::Difference => {
                // union other and interiors
                let mut current_clip: CompoundCurve<T, U3> = other.clone();
                let mut interiors = vec![];

                for interior in self.interiors() {
                    let union =
                        current_clip.boolean(BooleanOperation::Union, interior, option.clone())?;
                    let mut regions_iter = union.into_regions().into_iter();
                    let head = regions_iter.next();
                    if let Some(head) = head {
                        current_clip = head.into_exterior();
                    }
                    regions_iter.for_each(|region| interiors.push(region.into_exterior()));
                }

                let clip = self.exterior().boolean(
                    BooleanOperation::Difference,
                    &current_clip,
                    option.clone(),
                )?;
                let mut regions = clip.into_regions();
                distribute_interiors_to_regions(&mut regions, &mut interiors, option.clone())?;
                Ok(Clip::new(regions, Default::default()))
            }
        }
    }
}

/// distribute the interiors into the regions that contain them
fn distribute_interiors_to_regions<T: FloatingPoint + ArgminFloat>(
    regions: &mut Vec<Region<T>>,
    interiors: &mut Vec<CompoundCurve<T, U3>>,
    option: Option<CurveIntersectionSolverOptions<T>>,
) -> anyhow::Result<()> {
    for region in regions.iter_mut() {
        let n = interiors.len();
        for i in (0..n).rev() {
            let pt = interiors[i].first_point();
            match region.exterior().contains(&pt, option.clone()) {
                Ok(true) => {
                    let it = interiors.remove(i);
                    region.interiors_mut().push(it);
                }
                Ok(false) => {}
                Err(e) => {
                    return Err(e);
                }
            }
        }
    }
    Ok(())
}
